import ctypes
from .base import _LIB, check_call
def set_bulk_size(size):
    """Set size limit on bulk execution.

    Bulk execution bundles many operators to run together.
    This can improve performance when running a lot of small
    operators sequentially.

    Parameters
    ----------
    size : int
        Maximum number of operators that can be bundled in a bulk.

    Returns
    -------
    int
        Previous bulk size.
    """
    prev = ctypes.c_int()
    check_call(_LIB.MXEngineSetBulkSize(ctypes.c_int(size), ctypes.byref(prev)))
    return prev.value