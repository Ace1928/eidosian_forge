import numpy
import cupy
from cupy._core import _routines_math as _math
from cupy._core import _fusion_thread_local
from cupy._core import internal
def ediff1d(arr, to_end=None, to_begin=None):
    """
    Calculates the difference between consecutive elements of an array.

    Args:
        arr (cupy.ndarray): Input array.
        to_end (cupy.ndarray, optional): Numbers to append at the end
            of the returend differences.
        to_begin (cupy.ndarray, optional): Numbers to prepend at the
            beginning of the returned differences.

    Returns:
        cupy.ndarray: New array consisting differences among succeeding
        elements.

    .. seealso:: :func:`numpy.ediff1d`
    """
    if not isinstance(arr, cupy.ndarray):
        raise TypeError('`arr` should be of type cupy.ndarray')
    arr = arr.ravel()
    dtype_req = arr.dtype
    if to_begin is None and to_end is None:
        return arr[1:] - arr[:-1]
    if to_begin is None:
        l_begin = 0
    else:
        if not isinstance(to_begin, cupy.ndarray):
            raise TypeError('`to_begin` should be of type cupy.ndarray')
        if not cupy.can_cast(to_begin, dtype_req, casting='same_kind'):
            raise TypeError('dtype of `to_begin` must be compatible with input `arr` under the `same_kind` rule.')
        to_begin = to_begin.ravel()
        l_begin = len(to_begin)
    if to_end is None:
        l_end = 0
    else:
        if not isinstance(to_end, cupy.ndarray):
            raise TypeError('`to_end` should be of type cupy.ndarray')
        if not cupy.can_cast(to_end, dtype_req, casting='same_kind'):
            raise TypeError('dtype of `to_end` must be compatible with input `arr` under the `same_kind` rule.')
        to_end = to_end.ravel()
        l_end = len(to_end)
    l_diff = max(len(arr) - 1, 0)
    result = cupy.empty(l_diff + l_begin + l_end, dtype=arr.dtype)
    if l_begin > 0:
        result[:l_begin] = to_begin
    if l_end > 0:
        result[l_begin + l_diff:] = to_end
    cupy.subtract(arr[1:], arr[:-1], result[l_begin:l_begin + l_diff])
    return result