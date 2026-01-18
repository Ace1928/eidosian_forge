import warnings
import cupy
from cupy import _core
from cupy._core import _routines_statistics as _statistics
from cupy._core import _fusion_thread_local
from cupy._logic import content
def nanmin(a, axis=None, out=None, keepdims=False):
    """Returns the minimum of an array along an axis ignoring NaN.

    When there is a slice whose elements are all NaN, a :class:`RuntimeWarning`
    is raised and NaN is returned.

    Args:
        a (cupy.ndarray): Array to take the minimum.
        axis (int): Along which axis to take the minimum. The flattened array
            is used by default.
        out (cupy.ndarray): Output array.
        keepdims (bool): If ``True``, the axis is remained as an axis of
            size one.

    Returns:
        cupy.ndarray: The minimum of ``a``, along the axis if specified.

    .. warning::

        This function may synchronize the device.

    .. seealso:: :func:`numpy.nanmin`

    """
    res = _core.nanmin(a, axis=axis, out=out, keepdims=keepdims)
    if content.isnan(res).any():
        warnings.warn('All-NaN slice encountered', RuntimeWarning)
    return res