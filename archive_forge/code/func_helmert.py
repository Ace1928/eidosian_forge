import math
import cupy
from cupy import _core
from cupyx.scipy.linalg import _uarray
@_uarray.implements('helmert')
def helmert(n, full=False):
    """Create an Helmert matrix of order ``n``.

    This has applications in statistics, compositional or simplicial analysis,
    and in Aitchison geometry.

    Args:
        n (int): The size of the array to create.
        full (bool, optional): If True the (n, n) ndarray will be returned.
            Otherwise, the default, the submatrix that does not include the
            first row will be returned.

    Returns:
        cupy.ndarray: The Helmert matrix. The shape is (n, n) or (n-1, n)
        depending on the ``full`` argument.

    .. seealso:: :func:`scipy.linalg.helmert`
    """
    d = cupy.arange(n)
    H = cupy.tri(n, n, -1)
    H.diagonal()[:] -= d
    d *= cupy.arange(1, n + 1)
    H[0] = 1
    d[0] = n
    H /= cupy.sqrt(d)[:, None]
    return H if full else H[1:]