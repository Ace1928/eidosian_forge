import numpy as np
import scipy.special
import scipy.special._ufuncs as scu
from scipy._lib._finite_differences import _derivative
def kolmognp(n, x):
    """Computes the PDF for the two-sided Kolmogorov-Smirnov distribution.

    Parameters
    ----------
    n : integer, array_like
        the number of samples
    x : float, array_like
        The K-S statistic, float between 0 and 1

    Returns
    -------
    pdf : ndarray
        The PDF at the specified locations

    The return value has shape the result of numpy broadcasting n and x.
    """
    it = np.nditer([n, x, None])
    for _n, _x, z in it:
        if np.isnan(_n):
            z[...] = _n
            continue
        if int(_n) != _n:
            raise ValueError(f'n is not integral: {_n}')
        z[...] = _kolmogn_p(int(_n), _x)
    result = it.operands[-1]
    return result