import numpy as np
from statsmodels.tools.tools import Bunch
def cc_ranktest(x1, x2, demean=True, fullrank=False):
    """rank tests based on smallest canonical correlation coefficients

    Anderson canonical correlations test (LM test) and
    Cragg-Donald test (Wald test)
    Assumes homoskedasticity and independent observations, overrejects if
    there is heteroscedasticity or autocorrelation.

    The Null Hypothesis is that the rank is k - 1, the alternative hypothesis
    is that the rank is at least k.


    Parameters
    ----------
    x1, x2 : ndarrays, 2_D
        two 2-dimensional data arrays, observations in rows, variables in columns
    demean : bool
         If demean is true, then the mean is subtracted from each variable.
    fullrank : bool
         If true, then only the test that the matrix has full rank is returned.
         If false, the test for all possible ranks are returned. However, no
         the p-values are not corrected for the multiplicity of tests.

    Returns
    -------
    value : float
        value of the test statistic
    p-value : float
        p-value for the test Null Hypothesis tha the smallest canonical
        correlation coefficient is zero. based on chi-square distribution
    df : int
        degrees of freedom for thechi-square distribution in the hypothesis test
    ccorr : ndarray, 1d
        All canonical correlation coefficients sorted from largest to smallest.

    Notes
    -----
    Degrees of freedom for the distribution of the test statistic are based on
    number of columns of x1 and x2 and not on their matrix rank.
    (I'm not sure yet what the interpretation of the test is if x1 or x2 are of
    reduced rank.)

    See Also
    --------
    cancorr
    cc_stats

    """
    from scipy import stats
    nobs1, k1 = x1.shape
    nobs2, k2 = x2.shape
    cc = cancorr(x1, x2, demean=demean)
    cc2 = cc * cc
    if fullrank:
        df = np.abs(k1 - k2) + 1
        value = nobs1 * cc2[-1]
        w_value = nobs1 * (cc2[-1] / (1.0 - cc2[-1]))
        return (value, stats.chi2.sf(value, df), df, cc, w_value, stats.chi2.sf(w_value, df))
    else:
        r = np.arange(min(k1, k2))[::-1]
        df = (k1 - r) * (k2 - r)
        values = nobs1 * cc2[::-1].cumsum()
        w_values = nobs1 * (cc2 / (1.0 - cc2))[::-1].cumsum()
        return (values, stats.chi2.sf(values, df), df, cc, w_values, stats.chi2.sf(w_values, df))