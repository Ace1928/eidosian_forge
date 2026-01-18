import numpy as np
import numpy.linalg as la
from numpy.core.multiarray import normalize_axis_index
from . import polyutils as pu
from ._polybase import ABCPolyBase
def hermder(c, m=1, scl=1, axis=0):
    """
    Differentiate a Hermite series.

    Returns the Hermite series coefficients `c` differentiated `m` times
    along `axis`.  At each iteration the result is multiplied by `scl` (the
    scaling factor is for use in a linear change of variable). The argument
    `c` is an array of coefficients from low to high degree along each
    axis, e.g., [1,2,3] represents the series ``1*H_0 + 2*H_1 + 3*H_2``
    while [[1,2],[1,2]] represents ``1*H_0(x)*H_0(y) + 1*H_1(x)*H_0(y) +
    2*H_0(x)*H_1(y) + 2*H_1(x)*H_1(y)`` if axis=0 is ``x`` and axis=1 is
    ``y``.

    Parameters
    ----------
    c : array_like
        Array of Hermite series coefficients. If `c` is multidimensional the
        different axis correspond to different variables with the degree in
        each axis given by the corresponding index.
    m : int, optional
        Number of derivatives taken, must be non-negative. (Default: 1)
    scl : scalar, optional
        Each differentiation is multiplied by `scl`.  The end result is
        multiplication by ``scl**m``.  This is for use in a linear change of
        variable. (Default: 1)
    axis : int, optional
        Axis over which the derivative is taken. (Default: 0).

        .. versionadded:: 1.7.0

    Returns
    -------
    der : ndarray
        Hermite series of the derivative.

    See Also
    --------
    hermint

    Notes
    -----
    In general, the result of differentiating a Hermite series does not
    resemble the same operation on a power series. Thus the result of this
    function may be "unintuitive," albeit correct; see Examples section
    below.

    Examples
    --------
    >>> from numpy.polynomial.hermite import hermder
    >>> hermder([ 1. ,  0.5,  0.5,  0.5])
    array([1., 2., 3.])
    >>> hermder([-0.5,  1./2.,  1./8.,  1./12.,  1./16.], m=2)
    array([1., 2., 3.])

    """
    c = np.array(c, ndmin=1, copy=True)
    if c.dtype.char in '?bBhHiIlLqQpP':
        c = c.astype(np.double)
    cnt = pu._deprecate_as_int(m, 'the order of derivation')
    iaxis = pu._deprecate_as_int(axis, 'the axis')
    if cnt < 0:
        raise ValueError('The order of derivation must be non-negative')
    iaxis = normalize_axis_index(iaxis, c.ndim)
    if cnt == 0:
        return c
    c = np.moveaxis(c, iaxis, 0)
    n = len(c)
    if cnt >= n:
        c = c[:1] * 0
    else:
        for i in range(cnt):
            n = n - 1
            c *= scl
            der = np.empty((n,) + c.shape[1:], dtype=c.dtype)
            for j in range(n, 0, -1):
                der[j - 1] = 2 * j * c[j]
            c = der
    c = np.moveaxis(c, 0, iaxis)
    return c