import math
import cupy
from cupy import _core
from cupyx.scipy.linalg import _uarray
@_uarray.implements('leslie')
def leslie(f, s):
    """Create a Leslie matrix.

    Given the length n array of fecundity coefficients ``f`` and the length n-1
    array of survival coefficients ``s``, return the associated Leslie matrix.

    Args:
        f (cupy.ndarray): The "fecundity" coefficients.
        s (cupy.ndarray): The "survival" coefficients, has to be 1-D.  The
            length of ``s`` must be one less than the length of ``f``, and it
            must be at least 1.

    Returns:
        cupy.ndarray: The array is zero except for the first row, which is
        ``f``, and the first sub-diagonal, which is ``s``. The data-type of
        the array will be the data-type of ``f[0]+s[0]``.

    .. seealso:: :func:`scipy.linalg.leslie`
    """
    if f.ndim != 1:
        raise ValueError('Incorrect shape for f. f must be 1D')
    if s.ndim != 1:
        raise ValueError('Incorrect shape for s. s must be 1D')
    n = f.size
    if n != s.size + 1:
        raise ValueError('Length of s must be one less than length of f')
    if s.size == 0:
        raise ValueError('The length of s must be at least 1.')
    a = cupy.zeros((n, n), dtype=cupy.result_type(f, s))
    a[0] = f
    cupy.fill_diagonal(a[1:], s)
    return a