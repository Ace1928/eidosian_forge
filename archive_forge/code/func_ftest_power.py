import warnings
import numpy as np
from scipy import stats, optimize, special
from statsmodels.tools.rootfinding import brentq_expanding
def ftest_power(effect_size, df2, df1, alpha, ncc=1):
    """Calculate the power of a F-test.

    Parameters
    ----------
    effect_size : float
        The effect size is here Cohen's ``f``, the square root of ``f2``.
    df2 : int or float
        Denominator degrees of freedom.
        This corresponds to the df_resid in Wald tests.
    df1 : int or float
        Numerator degrees of freedom.
        This corresponds to the number of constraints in Wald tests.
    alpha : float in interval (0,1)
        significance level, e.g. 0.05, is the probability of a type I
        error, that is wrong rejections if the Null Hypothesis is true.
    ncc : int
        degrees of freedom correction for non-centrality parameter.
        see Notes

    Returns
    -------
    power : float
        Power of the test, e.g. 0.8, is one minus the probability of a
        type II error. Power is the probability that the test correctly
        rejects the Null Hypothesis if the Alternative Hypothesis is true.

    Notes
    -----
    changed in 0.14: use df2, df1 instead of df_num, df_denom as arg names.
    The latter had reversed meaning.

    The sample size is given implicitly by ``df2`` with fixed number of
    constraints given by numerator degrees of freedom ``df1``:

        nobs = df2 + df1 + ncc

    Set ncc=0 to match t-test, or f-test in LikelihoodModelResults.
    ncc=1 matches the non-centrality parameter in R::pwr::pwr.f2.test

    ftest_power with ncc=0 should also be correct for f_test in regression
    models, with df_num (df1) as number of constraints and d_denom (df2) as
    df_resid.
    """
    df_num, df_denom = (df1, df2)
    nc = effect_size ** 2 * (df_denom + df_num + ncc)
    crit = stats.f.isf(alpha, df_num, df_denom)
    pow_ = ncf_sf(crit, df_num, df_denom, nc)
    return pow_