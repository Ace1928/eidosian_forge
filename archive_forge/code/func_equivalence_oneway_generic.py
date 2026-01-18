import numpy as np
from scipy import stats
from scipy.special import ncfdtrinc
from statsmodels.stats.power import ncf_cdf, ncf_ppf
from statsmodels.stats.robust_compare import TrimmedMean, scale_transform
from statsmodels.tools.testing import Holder
from statsmodels.stats.base import HolderTuple
def equivalence_oneway_generic(f_stat, n_groups, nobs, equiv_margin, df, alpha=0.05, margin_type='f2'):
    """Equivalence test for oneway anova (Wellek and extensions)

    This is an helper function when summary statistics are available.
    Use `equivalence_oneway` instead.

    The null hypothesis is that the means differ by more than `equiv_margin`
    in the anova distance measure.
    If the Null is rejected, then the data supports that means are equivalent,
    i.e. within a given distance.

    Parameters
    ----------
    f_stat : float
        F-statistic
    n_groups : int
        Number of groups in oneway comparison.
    nobs : ndarray
        Array of number of observations in groups.
    equiv_margin : float
        Equivalence margin in terms of effect size. Effect size can be chosen
        with `margin_type`. default is squared Cohen's f.
    df : tuple
        degrees of freedom ``df = (df1, df2)`` where

        - df1 : numerator degrees of freedom, number of constraints
        - df2 : denominator degrees of freedom, df_resid

    alpha : float in (0, 1)
        Significance level for the hypothesis test.
    margin_type : "f2" or "wellek"
        Type of effect size used for equivalence margin.

    Returns
    -------
    results : instance of HolderTuple class
        The two main attributes are test statistic `statistic` and p-value
        `pvalue`.

    Notes
    -----
    Equivalence in this function is defined in terms of a squared distance
    measure similar to Mahalanobis distance.
    Alternative definitions for the oneway case are based on maximum difference
    between pairs of means or similar pairwise distances.

    The equivalence margin is used for the noncentrality parameter in the
    noncentral F distribution for the test statistic. In samples with unequal
    variances estimated using Welch or Brown-Forsythe Anova, the f-statistic
    depends on the unequal variances and corrections to the test statistic.
    This means that the equivalence margins are not fully comparable across
    methods for treating unequal variances.

    References
    ----------
    Wellek, Stefan. 2010. Testing Statistical Hypotheses of Equivalence and
    Noninferiority. 2nd ed. Boca Raton: CRC Press.

    Cribbie, Robert A., Chantal A. Arpin-Cribbie, and Jamie A. Gruman. 2009.
    “Tests of Equivalence for One-Way Independent Groups Designs.” The Journal
    of Experimental Education 78 (1): 1–13.
    https://doi.org/10.1080/00220970903224552.

    Jan, Show-Li, and Gwowen Shieh. 2019. “On the Extended Welch Test for
    Assessing Equivalence of Standardized Means.” Statistics in
    Biopharmaceutical Research 0 (0): 1–8.
    https://doi.org/10.1080/19466315.2019.1654915.

    """
    nobs_t = nobs.sum()
    nobs_mean = nobs_t / n_groups
    if margin_type == 'wellek':
        nc_null = nobs_mean * equiv_margin ** 2
        es = f_stat * (n_groups - 1) / nobs_mean
        type_effectsize = "Wellek's psi_squared"
    elif margin_type in ['f2', 'fsqu', 'fsquared']:
        nc_null = nobs_t * equiv_margin
        es = f_stat / nobs_t
        type_effectsize = "Cohen's f_squared"
    else:
        raise ValueError('`margin_type` should be "f2" or "wellek"')
    crit_f = ncf_ppf(alpha, df[0], df[1], nc_null)
    if margin_type == 'wellek':
        crit_es = crit_f * (n_groups - 1) / nobs_mean
    elif margin_type in ['f2', 'fsqu', 'fsquared']:
        crit_es = crit_f / nobs_t
    reject = es < crit_es
    pv = ncf_cdf(f_stat, df[0], df[1], nc_null)
    pwr = ncf_cdf(crit_f, df[0], df[1], 1e-13)
    res = HolderTuple(statistic=f_stat, pvalue=pv, effectsize=es, crit_f=crit_f, crit_es=crit_es, reject=reject, power_zero=pwr, df=df, f_stat=f_stat, type_effectsize=type_effectsize)
    return res