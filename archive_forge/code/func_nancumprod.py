import numpy
import cupy
from cupy._core import _routines_math as _math
from cupy._core import _fusion_thread_local
from cupy._core import internal
def nancumprod(a, axis=None, dtype=None, out=None):
    """Returns the cumulative product of an array along a given axis treating
    Not a Numbers (NaNs) as one.

    Args:
        a (cupy.ndarray): Input array.
        axis (int): Axis along which the cumulative product is taken. If it is
            not specified, the input is flattened.
        dtype: Data type specifier.
        out (cupy.ndarray): Output array.

    Returns:
        cupy.ndarray: The result array.

    .. seealso:: :func:`numpy.nancumprod`
    """
    a = _replace_nan(a, 1, out=out)
    return cumprod(a, axis=axis, dtype=dtype, out=out)