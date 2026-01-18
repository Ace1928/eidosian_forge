import cupy
from cupy._core import _routines_logic as _logic
from cupy._core import _fusion_thread_local
from cupy._sorting import search as _search
from cupy import _util
def in1d(ar1, ar2, assume_unique=False, invert=False):
    """Tests whether each element of a 1-D array is also present in a second
    array.

    Returns a boolean array the same length as ``ar1`` that is ``True``
    where an element of ``ar1`` is in ``ar2`` and ``False`` otherwise.

    Parameters
    ----------
    ar1 : cupy.ndarray
        Input array.
    ar2 : cupy.ndarray
        The values against which to test each value of ``ar1``.
    assume_unique : bool, optional
        Ignored
    invert : bool, optional
        If ``True``, the values in the returned array
        are inverted (that is, ``False`` where an element of ``ar1`` is in
        ``ar2`` and ``True`` otherwise). Default is ``False``.

    Returns
    -------
    y : cupy.ndarray, bool
        The values ``ar1[in1d]`` are in ``ar2``.

    """
    ar1 = ar1.ravel()
    ar2 = ar2.ravel()
    if ar1.size == 0 or ar2.size == 0:
        if invert:
            return cupy.ones(ar1.shape, dtype=cupy.bool_)
        else:
            return cupy.zeros(ar1.shape, dtype=cupy.bool_)
    ar2 = cupy.sort(ar2)
    return _search._exists_kernel(ar1, ar2, ar2.size, invert)