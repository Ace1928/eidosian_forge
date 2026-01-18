import math
import cupy
from cupy import _core
def multigammaln(a, d):
    """Returns the log of multivariate gamma, also sometimes called the
    generalized gamma.

    Parameters
    ----------
    a : cupy.ndarray
        The multivariate gamma is computed for each item of `a`.
    d : int
        The dimension of the space of integration.

    Returns
    -------
    res : ndarray
        The values of the log multivariate gamma at the given points `a`.

    See Also
    --------
    :func:`scipy.special.multigammaln`

    """
    if not cupy.isscalar(d) or math.floor(d) != d:
        raise ValueError('d should be a positive integer (dimension)')
    if cupy.isscalar(a):
        a = cupy.asarray(a, dtype=float)
    if int(cupy.any(a <= 0.5 * (d - 1))):
        raise ValueError('condition a > 0.5 * (d-1) not met')
    res = d * (d - 1) * 0.25 * math.log(math.pi)
    gam0 = gammaln(a)
    if a.dtype.kind != 'f':
        gam0 = gam0.astype(cupy.float64)
    res = res + gam0
    for j in range(2, d + 1):
        res += gammaln(a - (j - 1.0) / 2)
    return res