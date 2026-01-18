import warnings
import numpy as np
from scipy import stats
from statsmodels.tools.validation import array_like, bool_like, int_like
def normal_ad(x, axis=0):
    """
    Anderson-Darling test for normal distribution unknown mean and variance.

    Parameters
    ----------
    x : array_like
        The data array.
    axis : int
        The axis to perform the test along.

    Returns
    -------
    ad2 : float
        Anderson Darling test statistic.
    pval : float
        The pvalue for hypothesis that the data comes from a normal
        distribution with unknown mean and variance.

    See Also
    --------
    statsmodels.stats.diagnostic.anderson_statistic
        The Anderson-Darling a2 statistic.
    statsmodels.stats.diagnostic.kstest_fit
        Kolmogorov-Smirnov test with estimated parameters for Normal or
        Exponential distributions.
    """
    ad2 = anderson_statistic(x, dist='norm', fit=True, axis=axis)
    n = x.shape[axis]
    ad2a = ad2 * (1 + 0.75 / n + 2.25 / n ** 2)
    if np.size(ad2a) == 1:
        if ad2a >= 0.0 and ad2a < 0.2:
            pval = 1 - np.exp(-13.436 + 101.14 * ad2a - 223.73 * ad2a ** 2)
        elif ad2a < 0.34:
            pval = 1 - np.exp(-8.318 + 42.796 * ad2a - 59.938 * ad2a ** 2)
        elif ad2a < 0.6:
            pval = np.exp(0.9177 - 4.279 * ad2a - 1.38 * ad2a ** 2)
        elif ad2a <= 13:
            pval = np.exp(1.2937 - 5.709 * ad2a + 0.0186 * ad2a ** 2)
        else:
            pval = 0.0
    else:
        bounds = np.array([0.0, 0.2, 0.34, 0.6])
        pval0 = lambda ad2a: np.nan * np.ones_like(ad2a)
        pval1 = lambda ad2a: 1 - np.exp(-13.436 + 101.14 * ad2a - 223.73 * ad2a ** 2)
        pval2 = lambda ad2a: 1 - np.exp(-8.318 + 42.796 * ad2a - 59.938 * ad2a ** 2)
        pval3 = lambda ad2a: np.exp(0.9177 - 4.279 * ad2a - 1.38 * ad2a ** 2)
        pval4 = lambda ad2a: np.exp(1.2937 - 5.709 * ad2a + 0.0186 * ad2a ** 2)
        pvalli = [pval0, pval1, pval2, pval3, pval4]
        idx = np.searchsorted(bounds, ad2a, side='right')
        pval = np.nan * np.ones_like(ad2a)
        for i in range(5):
            mask = idx == i
            pval[mask] = pvalli[i](ad2a[mask])
    return (ad2, pval)