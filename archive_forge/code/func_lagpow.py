import numpy as np
import numpy.linalg as la
from numpy.core.multiarray import normalize_axis_index
from . import polyutils as pu
from ._polybase import ABCPolyBase
def lagpow(c, pow, maxpower=16):
    """Raise a Laguerre series to a power.

    Returns the Laguerre series `c` raised to the power `pow`. The
    argument `c` is a sequence of coefficients ordered from low to high.
    i.e., [1,2,3] is the series  ``P_0 + 2*P_1 + 3*P_2.``

    Parameters
    ----------
    c : array_like
        1-D array of Laguerre series coefficients ordered from low to
        high.
    pow : integer
        Power to which the series will be raised
    maxpower : integer, optional
        Maximum power allowed. This is mainly to limit growth of the series
        to unmanageable size. Default is 16

    Returns
    -------
    coef : ndarray
        Laguerre series of power.

    See Also
    --------
    lagadd, lagsub, lagmulx, lagmul, lagdiv

    Examples
    --------
    >>> from numpy.polynomial.laguerre import lagpow
    >>> lagpow([1, 2, 3], 2)
    array([ 14., -16.,  56., -72.,  54.])

    """
    return pu._pow(lagmul, c, pow, maxpower)