import math
import cupy
from cupy import _core
from cupyx.scipy.linalg import _uarray
@_uarray.implements('companion')
def companion(a):
    """Create a companion matrix.

    Create the companion matrix associated with the polynomial whose
    coefficients are given in ``a``.

    Args:
        a (cupy.ndarray): 1-D array of polynomial coefficients. The length of
            ``a`` must be at least two, and ``a[0]`` must not be zero.

    Returns:
        (cupy.ndarray): The first row of the output is ``-a[1:]/a[0]``, and the
        first sub-diagonal is all ones. The data-type of the array is the
        same as the data-type of ``-a[1:]/a[0]``.

    .. seealso:: :func:`cupyx.scipy.linalg.fiedler_companion`
    .. seealso:: :func:`scipy.linalg.companion`
    """
    n = a.size
    if a.ndim != 1:
        raise ValueError('`a` must be one-dimensional.')
    if n < 2:
        raise ValueError('The length of `a` must be at least 2.')
    first_row = -a[1:] / a[0]
    c = cupy.zeros((n - 1, n - 1), dtype=first_row.dtype)
    c[0] = first_row
    cupy.fill_diagonal(c[1:], 1)
    return c