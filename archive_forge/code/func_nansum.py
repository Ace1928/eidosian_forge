import numpy
import cupy
from cupy._core import _routines_math as _math
from cupy._core import _fusion_thread_local
from cupy._core import internal
def nansum(a, axis=None, dtype=None, out=None, keepdims=False):
    """Returns the sum of an array along given axes treating Not a Numbers
    (NaNs) as zero.

    Args:
        a (cupy.ndarray): Array to take sum.
        axis (int or sequence of ints): Axes along which the sum is taken.
        dtype: Data type specifier.
        out (cupy.ndarray): Output array.
        keepdims (bool): If ``True``, the specified axes are remained as axes
            of length one.

    Returns:
        cupy.ndarray: The result array.

    .. seealso:: :func:`numpy.nansum`

    """
    if _fusion_thread_local.is_fusing():
        if keepdims:
            raise NotImplementedError('cupy.nansum does not support `keepdims` in fusion yet.')
        if a.dtype.char in 'FD':
            func = _math._nansum_complex_dtype
        elif dtype is None:
            func = _math._nansum_auto_dtype
        else:
            func = _math._nansum_keep_dtype
        return _fusion_thread_local.call_reduction(func, a, axis=axis, dtype=dtype, out=out)
    return _math._nansum(a, axis, dtype, out, keepdims)