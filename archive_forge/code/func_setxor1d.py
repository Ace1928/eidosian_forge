import cupy
from cupy._core import _routines_logic as _logic
from cupy._core import _fusion_thread_local
from cupy._sorting import search as _search
from cupy import _util
def setxor1d(ar1, ar2, assume_unique=False):
    """Find the set exclusive-or of two arrays.

    Parameters
    ----------
    ar1, ar2 : cupy.ndarray
        Input arrays. They are flattend if they are not already 1-D.
    assume_unique : bool
        By default, False, i.e. input arrays are not unique.
        If True, input arrays are assumed to be unique. This can
        speed up the calculation.

    Returns
    -------
    setxor1d : cupy.ndarray
        Return the sorted, unique values that are in only one
        (not both) of the input arrays.

    See Also
    --------
    numpy.setxor1d

    """
    if not assume_unique:
        ar1 = cupy.unique(ar1)
        ar2 = cupy.unique(ar2)
    aux = cupy.concatenate((ar1, ar2), axis=None)
    if aux.size == 0:
        return aux
    aux.sort()
    return aux[_setxorkernel(aux, aux.size, cupy.zeros(aux.size, dtype=cupy.bool_))]