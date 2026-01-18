import numpy as np
import numpy.linalg as la
from numpy.core.multiarray import normalize_axis_index
from . import polyutils as pu
from ._polybase import ABCPolyBase
def legmulx(c):
    """Multiply a Legendre series by x.

    Multiply the Legendre series `c` by x, where x is the independent
    variable.


    Parameters
    ----------
    c : array_like
        1-D array of Legendre series coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Array representing the result of the multiplication.

    See Also
    --------
    legadd, legmul, legdiv, legpow

    Notes
    -----
    The multiplication uses the recursion relationship for Legendre
    polynomials in the form

    .. math::

      xP_i(x) = ((i + 1)*P_{i + 1}(x) + i*P_{i - 1}(x))/(2i + 1)

    Examples
    --------
    >>> from numpy.polynomial import legendre as L
    >>> L.legmulx([1,2,3])
    array([ 0.66666667, 2.2, 1.33333333, 1.8]) # may vary

    """
    [c] = pu.as_series([c])
    if len(c) == 1 and c[0] == 0:
        return c
    prd = np.empty(len(c) + 1, dtype=c.dtype)
    prd[0] = c[0] * 0
    prd[1] = c[0]
    for i in range(1, len(c)):
        j = i + 1
        k = i - 1
        s = i + j
        prd[j] = c[i] * j / s
        prd[k] += c[i] * i / s
    return prd