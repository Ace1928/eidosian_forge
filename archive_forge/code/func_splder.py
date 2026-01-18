import operator
import cupy
from cupy._core import internal
from cupy._core._scalar import get_typename
from cupyx.scipy.sparse import csr_matrix
import numpy as np
def splder(tck, n=1):
    """
    Compute the spline representation of the derivative of a given spline

    Parameters
    ----------
    tck : tuple of (t, c, k)
        Spline whose derivative to compute
    n : int, optional
        Order of derivative to evaluate. Default: 1

    Returns
    -------
    tck_der : tuple of (t2, c2, k2)
        Spline of order k2=k-n representing the derivative
        of the input spline.

    Notes
    -----
    .. seealso:: :class:`scipy.interpolate.splder`

    See Also
    --------
    splantider, splev, spalde
    """
    if n < 0:
        return splantider(tck, -n)
    t, c, k = tck
    if n > k:
        raise ValueError('Order of derivative (n = %r) must be <= order of spline (k = %r)' % (n, tck[2]))
    sh = (slice(None),) + (None,) * len(c.shape[1:])
    try:
        for j in range(n):
            dt = t[k + 1:-1] - t[1:-k - 1]
            dt = dt[sh]
            c = (c[1:-1 - k] - c[:-2 - k]) * k / dt
            c = cupy.r_[c, np.zeros((k,) + c.shape[1:])]
            t = t[1:-1]
            k -= 1
    except FloatingPointError as e:
        raise ValueError('The spline has internal repeated knots and is not differentiable %d times' % n) from e
    return (t, c, k)