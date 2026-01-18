import numpy as np
import scipy.special
import scipy.special._ufuncs as scu
from scipy._lib._finite_differences import _derivative
def kolmogni(n, q, cdf=True):
    """Computes the PPF(or ISF) for the two-sided Kolmogorov-Smirnov distribution.

    Parameters
    ----------
    n : integer, array_like
        the number of samples
    q : float, array_like
        Probabilities, float between 0 and 1
    cdf : bool, optional
        whether to compute the PPF(default=true) or the ISF.

    Returns
    -------
    ppf : ndarray
        PPF (or ISF if cdf is False) at the specified locations

    The return value has shape the result of numpy broadcasting n and x.
    """
    it = np.nditer([n, q, cdf, None])
    for _n, _q, _cdf, z in it:
        if np.isnan(_n):
            z[...] = _n
            continue
        if int(_n) != _n:
            raise ValueError(f'n is not integral: {_n}')
        _pcdf, _psf = (_q, 1 - _q) if _cdf else (1 - _q, _q)
        z[...] = _kolmogni(int(_n), _pcdf, _psf)
    result = it.operands[-1]
    return result