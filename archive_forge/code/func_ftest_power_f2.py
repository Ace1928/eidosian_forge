import warnings
import numpy as np
from scipy import stats, optimize, special
from statsmodels.tools.rootfinding import brentq_expanding
def ftest_power_f2(effect_size, df_num, df_denom, alpha, ncc=1):
    """Calculate the power of a F-test.

    Based on Cohen's `f^2` effect size.

    This assumes

        df_num : numerator degrees of freedom, (number of constraints)
        df_denom : denominator degrees of freedom (df_resid in regression)
        nobs = df_denom + df_num + ncc
        nc = effect_size * nobs  (noncentrality index)

    Power is computed one-sided in the upper tail.

    Parameters
    ----------
    effect_size : float
        Cohen's f2 effect size or noncentrality divided by nobs.
    df_num : int or float
        Numerator degrees of freedom.
        This corresponds to the number of constraints in Wald tests.
    df_denom : int or float
        Denominator degrees of freedom.
        This corresponds to the df_resid in Wald tests.
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

    The sample size is given implicitly by ``df_denom`` with fixed number of
    constraints given by numerator degrees of freedom ``df_num``:

        nobs = df_denom + df_num + ncc

    Set ncc=0 to match t-test, or f-test in LikelihoodModelResults.
    ncc=1 matches the non-centrality parameter in R::pwr::pwr.f2.test

    ftest_power with ncc=0 should also be correct for f_test in regression
    models, with df_num (df1) as number of constraints and d_denom (df2) as
    df_resid.
    """
    nc = effect_size * (df_denom + df_num + ncc)
    crit = stats.f.isf(alpha, df_num, df_denom)
    pow_ = ncf_sf(crit, df_num, df_denom, nc)
    return pow_