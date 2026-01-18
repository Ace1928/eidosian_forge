import numpy as np
from scipy import stats
from scipy.stats import rankdata
from statsmodels.stats.base import HolderTuple
from statsmodels.stats.weightstats import (
def rank_compare_2ordinal(count1, count2, ddof=1, use_t=True):
    """
    Stochastically larger probability for 2 independent ordinal samples.

    This is a special case of `rank_compare_2indep` when the data are given as
    counts of two independent ordinal, i.e. ordered multinomial, samples.

    The statistic of interest is the probability that a random draw from the
    population of the first sample has a larger value than a random draw from
    the population of the second sample, specifically

        p = P(x1 > x2) + 0.5 * P(x1 = x2)

    Parameters
    ----------
    count1 : array_like
        Counts of the first sample, categories are assumed to be ordered.
    count2 : array_like
        Counts of the second sample, number of categories and ordering needs
        to be the same as for sample 1.
    ddof : scalar
        Degrees of freedom correction for variance estimation. The default
        ddof=1 corresponds to `rank_compare_2indep`.
    use_t : bool
        If use_t is true, the t distribution with Welch-Satterthwaite type
        degrees of freedom is used for p-value and confidence interval.
        If use_t is false, then the normal distribution is used.

    Returns
    -------
    res : RankCompareResult
        This includes methods for hypothesis tests and confidence intervals
        for the probability that sample 1 is stochastically larger than
        sample 2.

    See Also
    --------
    rank_compare_2indep
    RankCompareResult

    Notes
    -----
    The implementation is based on the appendix of Munzel and Hauschke (2003)
    with the addition of ``ddof`` so that the results match the general
    function `rank_compare_2indep`.

    """
    count1 = np.asarray(count1)
    count2 = np.asarray(count2)
    nobs1, nobs2 = (count1.sum(), count2.sum())
    freq1 = count1 / nobs1
    freq2 = count2 / nobs2
    cdf1 = np.concatenate(([0], freq1)).cumsum(axis=0)
    cdf2 = np.concatenate(([0], freq2)).cumsum(axis=0)
    cdfm1 = (cdf1[1:] + cdf1[:-1]) / 2
    cdfm2 = (cdf2[1:] + cdf2[:-1]) / 2
    prob1 = (cdfm2 * freq1).sum()
    prob2 = (cdfm1 * freq2).sum()
    var1 = (cdfm2 ** 2 * freq1).sum() - prob1 ** 2
    var2 = (cdfm1 ** 2 * freq2).sum() - prob2 ** 2
    var_prob = var1 / (nobs1 - ddof) + var2 / (nobs2 - ddof)
    nobs = nobs1 + nobs2
    var = nobs * var_prob
    vn1 = var1 * nobs2 * nobs1 / (nobs1 - ddof)
    vn2 = var2 * nobs1 * nobs2 / (nobs2 - ddof)
    df = (vn1 + vn2) ** 2 / (vn1 ** 2 / (nobs1 - 1) + vn2 ** 2 / (nobs2 - 1))
    res = RankCompareResult(statistic=None, pvalue=None, s1=None, s2=None, var1=var1, var2=var2, var=var, var_prob=var_prob, nobs1=nobs1, nobs2=nobs2, nobs=nobs, mean1=None, mean2=None, prob1=prob1, prob2=prob2, somersd1=prob1 * 2 - 1, somersd2=prob2 * 2 - 1, df=df, use_t=use_t)
    return res