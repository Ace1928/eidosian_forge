import numpy
import cupy
from cupy._core import _routines_math as _math
from cupy._core import _fusion_thread_local
from cupy._core import internal
def nanprod(a, axis=None, dtype=None, out=None, keepdims=False):
    """Returns the product of an array along given axes treating Not a Numbers
    (NaNs) as zero.

    Args:
        a (cupy.ndarray): Array to take product.
        axis (int or sequence of ints): Axes along which the product is taken.
        dtype: Data type specifier.
        out (cupy.ndarray): Output array.
        keepdims (bool): If ``True``, the specified axes are remained as axes
            of length one.

    Returns:
        cupy.ndarray: The result array.

    .. seealso:: :func:`numpy.nanprod`

    """
    if _fusion_thread_local.is_fusing():
        if keepdims:
            raise NotImplementedError('cupy.nanprod does not support `keepdims` in fusion yet.')
        if dtype is None:
            func = _math._nanprod_auto_dtype
        else:
            func = _math._nanprod_keep_dtype
        return _fusion_thread_local.call_reduction(func, a, axis=axis, dtype=dtype, out=out)
    return _math._nanprod(a, axis, dtype, out, keepdims)