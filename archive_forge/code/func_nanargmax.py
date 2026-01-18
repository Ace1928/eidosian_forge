import cupy
from cupy import _core
from cupy._core import fusion
from cupy import _util
from cupy._core import _routines_indexing as _indexing
from cupy._core import _routines_statistics as _statistics
def nanargmax(a, axis=None, dtype=None, out=None, keepdims=False):
    """Return the indices of the maximum values in the specified axis ignoring
    NaNs. For all-NaN slice ``-1`` is returned.
    Subclass cannot be passed yet, subok=True still unsupported

    Args:
        a (cupy.ndarray): Array to take nanargmax.
        axis (int): Along which axis to find the maximum. ``a`` is flattened by
            default.

    Returns:
        cupy.ndarray: The indices of the maximum of ``a``
        along an axis ignoring NaN values.

    .. note:: For performance reasons, ``cupy.nanargmax`` returns
            ``out of range values`` for all-NaN slice
            whereas ``numpy.nanargmax`` raises ``ValueError``
    .. seealso:: :func:`numpy.nanargmax`
    """
    if a.dtype.kind in 'biu':
        return argmax(a, axis=axis)
    return _statistics._nanargmax(a, axis, dtype, out, keepdims)