import numpy
import cupy
from cupy import _core
def diag_indices_from(arr):
    """
    Return the indices to access the main diagonal of an n-dimensional array.
    See `diag_indices` for full details.

    Args:
        arr (cupy.ndarray): At least 2-D.

    .. seealso:: :func:`numpy.diag_indices_from`

    """
    if not isinstance(arr, cupy.ndarray):
        raise TypeError('Argument must be cupy.ndarray')
    if not arr.ndim >= 2:
        raise ValueError('input array must be at least 2-d')
    if not cupy.all(cupy.diff(arr.shape) == 0):
        raise ValueError('All dimensions of input must be of equal length')
    return diag_indices(arr.shape[0], arr.ndim)