import numpy as np
import numpy.linalg as la
from numpy.core.multiarray import normalize_axis_index
from . import polyutils as pu
from ._polybase import ABCPolyBase
def chebadd(c1, c2):
    """
    Add one Chebyshev series to another.

    Returns the sum of two Chebyshev series `c1` + `c2`.  The arguments
    are sequences of coefficients ordered from lowest order term to
    highest, i.e., [1,2,3] represents the series ``T_0 + 2*T_1 + 3*T_2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of Chebyshev series coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Array representing the Chebyshev series of their sum.

    See Also
    --------
    chebsub, chebmulx, chebmul, chebdiv, chebpow

    Notes
    -----
    Unlike multiplication, division, etc., the sum of two Chebyshev series
    is a Chebyshev series (without having to "reproject" the result onto
    the basis set) so addition, just like that of "standard" polynomials,
    is simply "component-wise."

    Examples
    --------
    >>> from numpy.polynomial import chebyshev as C
    >>> c1 = (1,2,3)
    >>> c2 = (3,2,1)
    >>> C.chebadd(c1,c2)
    array([4., 4., 4.])

    """
    return pu._add(c1, c2)