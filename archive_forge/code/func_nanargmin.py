import cupy
from cupy import _core
from cupy._core import fusion
from cupy import _util
from cupy._core import _routines_indexing as _indexing
from cupy._core import _routines_statistics as _statistics
def nanargmin(a, axis=None, dtype=None, out=None, keepdims=False):
    """Return the indices of the minimum values in the specified axis ignoring
    NaNs. For all-NaN slice ``-1`` is returned.
    Subclass cannot be passed yet, subok=True still unsupported

    Args:
        a (cupy.ndarray): Array to take nanargmin.
        axis (int): Along which axis to find the minimum. ``a`` is flattened by
            default.

    Returns:
        cupy.ndarray: The indices of the minimum of ``a``
        along an axis ignoring NaN values.

    .. note:: For performance reasons, ``cupy.nanargmin`` returns
            ``out of range values`` for all-NaN slice
            whereas ``numpy.nanargmin`` raises ``ValueError``
    .. seealso:: :func:`numpy.nanargmin`
    """
    if a.dtype.kind in 'biu':
        return argmin(a, axis=axis)
    return _statistics._nanargmin(a, axis, dtype, out, keepdims)