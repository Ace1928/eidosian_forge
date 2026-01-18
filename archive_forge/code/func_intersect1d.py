import cupy
from cupy._core import _routines_logic as _logic
from cupy._core import _fusion_thread_local
from cupy._sorting import search as _search
from cupy import _util
def intersect1d(arr1, arr2, assume_unique=False, return_indices=False):
    """Find the intersection of two arrays.
    Returns the sorted, unique values that are in both of the input arrays.

    Parameters
    ----------
    arr1, arr2 : cupy.ndarray
        Input arrays. Arrays will be flattened if they are not in 1D.
    assume_unique : bool
        By default, False. If set True, the input arrays will be
        assumend to be unique, which speeds up the calculation. If set True,
        but the arrays are not unique, incorrect results and out-of-bounds
        indices could result.
    return_indices : bool
       By default, False. If True, the indices which correspond to the
       intersection of the two arrays are returned.

    Returns
    -------
    intersect1d : cupy.ndarray
        Sorted 1D array of common and unique elements.
    comm1 : cupy.ndarray
        The indices of the first occurrences of the common values
        in `arr1`. Only provided if `return_indices` is True.
    comm2 : cupy.ndarray
        The indices of the first occurrences of the common values
        in `arr2`. Only provided if `return_indices` is True.

    See Also
    --------
    numpy.intersect1d

    """
    if not assume_unique:
        if return_indices:
            arr1, ind1 = cupy.unique(arr1, return_index=True)
            arr2, ind2 = cupy.unique(arr2, return_index=True)
        else:
            arr1 = cupy.unique(arr1)
            arr2 = cupy.unique(arr2)
    else:
        arr1 = arr1.ravel()
        arr2 = arr2.ravel()
    if not return_indices:
        mask = _search._exists_kernel(arr1, arr2, arr2.size, False)
        return arr1[mask]
    mask, v1 = _search._exists_and_searchsorted_kernel(arr1, arr2, arr2.size, False)
    int1d = arr1[mask]
    arr1_indices = cupy.flatnonzero(mask)
    arr2_indices = v1[mask]
    if not assume_unique:
        arr1_indices = ind1[arr1_indices]
        arr2_indices = ind2[arr2_indices]
    return (int1d, arr1_indices, arr2_indices)