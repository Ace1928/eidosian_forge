import numpy as np
from scipy import stats
from scipy.special import ncfdtrinc
from statsmodels.stats.power import ncf_cdf, ncf_ppf
from statsmodels.stats.robust_compare import TrimmedMean, scale_transform
from statsmodels.tools.testing import Holder
from statsmodels.stats.base import HolderTuple
def confint_noncentrality(f_stat, df, alpha=0.05, alternative='two-sided'):
    """
    Confidence interval for noncentrality parameter in F-test

    This does not yet handle non-negativity constraint on nc.
    Currently only two-sided alternative is supported.

    Parameters
    ----------
    f_stat : float
    df : tuple
        degrees of freedom ``df = (df1, df2)`` where

        - df1 : numerator degrees of freedom, number of constraints
        - df2 : denominator degrees of freedom, df_resid

    alpha : float, default 0.05
    alternative : {"two-sided"}
        Other alternatives have not been implements.

    Returns
    -------
    float
        The end point of the confidence interval.

    Notes
    -----
    The algorithm inverts the cdf of the noncentral F distribution with
    respect to the noncentrality parameters.
    See Steiger 2004 and references cited in it.

    References
    ----------
    .. [1] Steiger, James H. 2004. “Beyond the F Test: Effect Size Confidence
       Intervals and Tests of Close Fit in the Analysis of Variance and
       Contrast Analysis.” Psychological Methods 9 (2): 164–82.
       https://doi.org/10.1037/1082-989X.9.2.164.

    See Also
    --------
    confint_effectsize_oneway
    """
    df1, df2 = df
    if alternative in ['two-sided', '2s', 'ts']:
        alpha1s = alpha / 2
        ci = ncfdtrinc(df1, df2, [1 - alpha1s, alpha1s], f_stat)
    else:
        raise NotImplementedError
    return ci