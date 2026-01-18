import warnings
import cupy
import numpy
from cupy.cuda import thrust
def msort(a):
    """Returns a copy of an array sorted along the first axis.

    Args:
        a (cupy.ndarray): Array to be sorted.

    Returns:
        cupy.ndarray: Array of the same type and shape as ``a``.

    .. note:
        ``cupy.msort(a)``, the CuPy counterpart of ``numpy.msort(a)``, is
        equivalent to ``cupy.sort(a, axis=0)``.

    .. seealso:: :func:`numpy.msort`

    """
    warnings.warn('msort is deprecated, use cupy.sort(a, axis=0) instead', DeprecationWarning)
    return sort(a, axis=0)