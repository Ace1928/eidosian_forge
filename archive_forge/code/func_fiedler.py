import math
import cupy
from cupy import _core
from cupyx.scipy.linalg import _uarray
@_uarray.implements('fiedler')
def fiedler(a):
    """Returns a symmetric Fiedler matrix

    Given an sequence of numbers ``a``, Fiedler matrices have the structure
    ``F[i, j] = np.abs(a[i] - a[j])``, and hence zero diagonals and nonnegative
    entries. A Fiedler matrix has a dominant positive eigenvalue and other
    eigenvalues are negative. Although not valid generally, for certain inputs,
    the inverse and the determinant can be derived explicitly.

    Args:
        a (cupy.ndarray): coefficient array

    Returns:
        cupy.ndarray: the symmetric Fiedler matrix

    .. seealso:: :func:`cupyx.scipy.linalg.circulant`
    .. seealso:: :func:`cupyx.scipy.linalg.toeplitz`
    .. seealso:: :func:`scipy.linalg.fiedler`
    """
    if a.ndim != 1:
        raise ValueError('Input `a` must be a 1D array.')
    if a.size == 0:
        return cupy.zeros(0)
    if a.size == 1:
        return cupy.zeros((1, 1))
    a = a[:, None] - a
    return cupy.abs(a, out=a)