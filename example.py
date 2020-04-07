
# # Example
#
# The following example demonstrates how to the use the `pymatch` package to match [Lending Club Loan Data](https://www.kaggle.com/wendykan/lending-club-loan-data). Follow the link to download the dataset from Kaggle (you'll have to create an account, it's fast and free!).
#
# Here we match Lending Club users that fully paid off loans (control) to those that defaulted (test). The example is contrived, however a use case for this could be that we want to analyze user sentiment with the platform. Users that default on loans may have worse sentiment because they are predisposed to a bad situation--influencing their perception of the product. Before analyzing sentiment, we can match users that paid their loans in full to users that defaulted based on the characteristics we can observe. If matching is successful, we could then make a statetment about the **causal effect** defaulting has on sentiment if we are confident our samples are sufficiently balanced and our model is free from omitted variable bias.
#
# This example, however, only goes through the matching procedure, which can be broken down into the following steps:
#
# * [Data Preparation](#Data-Prep)
# * [Fit Propensity Score Models](#Matcher)
# * [Predict Propensity Scores](#Predict-Scores)
# * [Tune Threshold](#Tune-Threshold)
# * [Match Data](#Match-Data)
# * [Assess Matches](#Assess-Matches)
#
# ----

# ### Data Prep


import warnings

warnings.filterwarnings("ignore")
from pymatch import Matcher
import pandas as pd
import numpy as np


# Load the dataset (`loan.csv`) and select a subset of columns.
#

# Create test and control groups and reassign `loan_status` to be a binary treatment indicator. This is our reponse in the logistic regression model(s) used to generate propensity scores.
data = pd.read_csv("./example_files/loan_sample.csv")
print(data.head())

test = data[data['loan_status'] == 1]
control = data[data['loan_status'] == 0]

# ----
#
# ### `Matcher`

# Initalize the `Matcher` object.
#
# **Note that:**
#
# * Upon intialization, `Matcher` prints the formula used to fit logistic regression model(s) and the number of records in the majority/minority class.
#     * The regression model(s) are used to generate propensity scores. In this case, we are using the covariates on the right side of the equation to estimate the probability of defaulting on a loan (`loan_status`= 1).
# * `Matcher` will use all covariates in the dataset unless a formula is specified by the user. Note that this step is only fitting model(s), we assign propensity scores later.
# * Any covariates passed to the (optional) `exclude` parameter will be ignored from the model fitting process. This parameter is particularly useful for unique identifiers like a `user_id`.


m = Matcher(test, control, yvar="loan_status", exclude=[])

# There is a significant imbalance in our data--the majority group (fully-paid loans) having many more records than the minority group (defaulted loans). We account for this by setting `balance=True` when calling `Matcher.fit_scores()` below. This tells `Matcher` to sample from the majority group when fitting the logistic regression model(s) so that the groups are of equal size. When undersampling this way, it is highly recommended that `nmodels` is explictly assigned to a integer much larger than 1. This ensure is that more of the majority group is contributing to the generation of propensity scores. The value of this integer should depend on the severity of the imbalance; here we use `nmodels`=100.


# for reproducibility
np.random.seed(20170925)

m.fit_scores(balance=True, nmodels=10)


# The average accuracy of our 100 models is 70.21%, suggesting that there's separability within our data and justifiying the need for the matching procedure. It's worth noting that we don't pay much attention to these logistic models since we are using them as a feature extraction tool (generation of propensity scores). The accuracy is a good way to detect separability at a glance, but we shouldn't spend time tuning and tinkering with these models. If our accuracy was close to 50%, that would suggest we cannot detect much separability in our groups given the features we observe and that matching is probably not necessary (or more features should be included if possible).

# ### Predict Scores


m.predict_scores()


m.plot_scores()

exit()

# The plot above demonstrates the separability present in our data. Test profiles have a much higher **propensity**, or estimated probability of defaulting given the features we isolated in the data.

# ---
#
# ### Tune Threshold

# The `Matcher.match()` method matches profiles that have propensity scores within some threshold.
#
# i.e. for two scores `s1` and `s2`, `|s1 - s2|` <= `threshold`
#
# By default matches are found *from* the majority group *for* the minority group. For example, if our test group contains 1,000 records and our control group contains 20,000, `Matcher` will
#     iterate through the test (minority) group and find suitable matches from the control (majority) group. If a record in the minority group has no suitable matches, it is dropped from the final matched dataset. We need to ensure our threshold is small enough such that we get close matches and retain most (or all) of our data in the minority group.
#
# Below we tune the threshold using `method="random"`. This matches a random profile that is within the threshold
# as there could be many. This is much faster than the alternative method "min", which finds the *closest* match for every minority record.


m.tune_threshold(method="random")


# It looks like a threshold of 0.0001 retains 100% of our data. Let's proceed with matching using this threshold.

# ---
#
# ### Match Data

# Below we match one record from the majority group to each record in the minority group. This is done **with** replacement, meaning a single majority record can be matched to multiple minority records. `Matcher` assigns a unique `record_id` to each record in the test and control groups so this can be addressed after matching. If susequent modelling is planned, one might consider weighting models using a weight vector of 1/`f` for each record, `f` being a record's frequency in the matched dataset. Thankfully `Matcher` can handle all of this for you :).


m.match(method="min", nmatches=1, threshold=0.0001)


m.record_frequency()


# It looks like the bulk of our matched-majority-group records occur only once, 68 occur twice, ... etc. We can preemptively generate a weight vector using `Matcher.assign_weight_vector()`


m.assign_weight_vector()


# Let's take a look at our matched data thus far. Note that in addition to the weight vector, `Matcher` has also assigned a `match_id` to each record indicating our (in this cased) *paired* matches since we use `nmatches=1`. We can verify that matched records have `scores` within 0.0001 of each other.


m.matched_data.sort_values("match_id").head(6)


# ---
#
# ### Assess Matches

# We must now determine if our data is "balanced". Can we detect any statistical differences between the covariates of our matched test and control groups? `Matcher` is configured to treat categorical and continouous variables separately in this assessment.

# ___Discrete___
#
# For categorical variables, we look at plots comparing the proportional differences between test and control before and after matching.
#
# For example, the first plot shows:
# * `prop_test` - `prop_control` for all possible `term` values---`prop_test` and `prop_control` being the proportion of test and control records with a given term value, respectively. We want these (orange) bars to be small after matching.
# * Results (pvalue) of a Chi-Square Test for Independence before and after matching. After matching we want this pvalue to be > 0.05, resulting in our failure to reject the null hypothesis that the frequecy of the enumerated term values are independent of our test and control groups.


categorical_results = m.compare_categorical(return_table=True)


categorical_results


# Looking at the plots and test results, we did a pretty good job balancing our categorical features! The p-values from the Chi-Square tests are all > 0.05 and we can verify by observing the small proportional differences in the plots.
#
# ___Continuous___
#
# For continous variables we look at Empirical Cumulative Distribution Functions (ECDF) for our test and control groups  before and after matching.
#
# For example, the first plot pair shows:
# * ECDF for test vs ECDF for control before matching (left), ECDF for test vs ECDF for control after matching(right). We want the two lines to be very close to each other (or indistiguishable) after matching.
# * Some tests + metrics are included in the chart titles.
#     * Tests performed:
#         * Kolmogorov-Smirnov Goodness of fit Test (KS-test)
#             This test statistic is calculated on 1000
#             permuted samples of the data, generating
#             an imperical p-value.  See pymatch.functions.ks_boot()
#             This is an adaptation of the ks.boot() method in
#             the R "Matching" package
#             https://www.rdocumentation.org/packages/Matching/versions/4.9-2/topics/ks.boot
#         * Chi-Square Distance:
#             Similarly this distance metric is calculated on
#             1000 permuted samples.
#             See pymatch.functions.grouped_permutation_test()
#
#     * Other included Stats:
#         * Standarized mean and median differences.
#              How many standard deviations away are the mean/median
#             between our groups before and after matching
#             i.e. `abs(mean(control) - mean(test))` / `std(control.union(test))`


cc = m.compare_continuous(return_table=True)


cc


# We want the pvalues from both the KS-test and the grouped permutation of the Chi-Square distance after matching to be > 0.05, and they all are! We can verify by looking at how close the ECDFs are between test and control.
#
# # Conclusion

# We saw a very "clean" result from the above procedure, achieving balance among all the covariates. In my work at Mozilla, we see much hairier results using the same procedure, which will likely be your experience too. In the case that certain covariates are not well balanced, one might consider tinkering with the parameters of the matching process (`nmatches`>1) or adding more covariates to the formula specified when we initialized the `Matcher` object.
# In any case, in subsequent modelling, you can always control for variables that you haven't deemed "balanced".
