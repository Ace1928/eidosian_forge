import numpy as np
import scipy.special
import scipy.special._ufuncs as scu
from scipy._lib._finite_differences import _derivative
def kolmogn(n, x, cdf=True):
    """Computes the CDF for the two-sided Kolmogorov-Smirnov distribution.

    The two-sided Kolmogorov-Smirnov distribution has as its CDF Pr(D_n <= x),
    for a sample of size n drawn from a distribution with CDF F(t), where
    D_n &= sup_t |F_n(t) - F(t)|, and
    F_n(t) is the Empirical Cumulative Distribution Function of the sample.

    Parameters
    ----------
    n : integer, array_like
        the number of samples
    x : float, array_like
        The K-S statistic, float between 0 and 1
    cdf : bool, optional
        whether to compute the CDF(default=true) or the SF.

    Returns
    -------
    cdf : ndarray
        CDF (or SF it cdf is False) at the specified locations.

    The return value has shape the result of numpy broadcasting n and x.
    """
    it = np.nditer([n, x, cdf, None], op_dtypes=[None, np.float64, np.bool_, np.float64])
    for _n, _x, _cdf, z in it:
        if np.isnan(_n):
            z[...] = _n
            continue
        if int(_n) != _n:
            raise ValueError(f'n is not integral: {_n}')
        z[...] = _kolmogn(int(_n), _x, cdf=_cdf)
    result = it.operands[-1]
    return result