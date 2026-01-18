import numpy as np
import numpy.linalg as la
from numpy.core.multiarray import normalize_axis_index
from . import polyutils as pu
from ._polybase import ABCPolyBase
def chebmul(c1, c2):
    """
    Multiply one Chebyshev series by another.

    Returns the product of two Chebyshev series `c1` * `c2`.  The arguments
    are sequences of coefficients, from lowest order "term" to highest,
    e.g., [1,2,3] represents the series ``T_0 + 2*T_1 + 3*T_2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of Chebyshev series coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Of Chebyshev series coefficients representing their product.

    See Also
    --------
    chebadd, chebsub, chebmulx, chebdiv, chebpow

    Notes
    -----
    In general, the (polynomial) product of two C-series results in terms
    that are not in the Chebyshev polynomial basis set.  Thus, to express
    the product as a C-series, it is typically necessary to "reproject"
    the product onto said basis set, which typically produces
    "unintuitive live" (but correct) results; see Examples section below.

    Examples
    --------
    >>> from numpy.polynomial import chebyshev as C
    >>> c1 = (1,2,3)
    >>> c2 = (3,2,1)
    >>> C.chebmul(c1,c2) # multiplication requires "reprojection"
    array([  6.5,  12. ,  12. ,   4. ,   1.5])

    """
    [c1, c2] = pu.as_series([c1, c2])
    z1 = _cseries_to_zseries(c1)
    z2 = _cseries_to_zseries(c2)
    prd = _zseries_mul(z1, z2)
    ret = _zseries_to_cseries(prd)
    return pu.trimseq(ret)