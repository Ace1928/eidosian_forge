import math
import cupy
from cupy import _core
from cupyx.scipy.linalg import _uarray
@_uarray.implements('circulant')
def circulant(c):
    """Construct a circulant matrix.

    Args:
        c (cupy.ndarray): 1-D array, the first column of the matrix.

    Returns:
        cupy.ndarray: A circulant matrix whose first column is ``c``.

    .. seealso:: :func:`cupyx.scipy.linalg.toeplitz`
    .. seealso:: :func:`cupyx.scipy.linalg.hankel`
    .. seealso:: :func:`cupyx.scipy.linalg.solve_circulant`
    .. seealso:: :func:`cupyx.scipy.linalg.fiedler`
    .. seealso:: :func:`scipy.linalg.circulant`
    """
    c = c.ravel()
    return _create_toeplitz_matrix(c[::-1], c[:0:-1])