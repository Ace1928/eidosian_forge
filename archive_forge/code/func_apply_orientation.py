import numpy as np
import numpy.linalg as npl
from .deprecated import deprecate_with_version
def apply_orientation(arr, ornt):
    """Apply transformations implied by `ornt` to the first
    n axes of the array `arr`

    Parameters
    ----------
    arr : array-like of data with ndim >= n
    ornt : (n,2) orientation array
       orientation transform. ``ornt[N,1]` is flip of axis N of the
       array implied by `shape`, where 1 means no flip and -1 means
       flip.  For example, if ``N==0`` and ``ornt[0,1] == -1``, and
       there's an array ``arr`` of shape `shape`, the flip would
       correspond to the effect of ``np.flipud(arr)``.  ``ornt[:,0]`` is
       the transpose that needs to be done to the implied array, as in
       ``arr.transpose(ornt[:,0])``

    Returns
    -------
    t_arr : ndarray
       data array `arr` transformed according to ornt
    """
    t_arr = np.asarray(arr)
    ornt = np.asarray(ornt)
    n = ornt.shape[0]
    if t_arr.ndim < n:
        raise OrientationError('Data array has fewer dimensions than orientation')
    if np.any(np.isnan(ornt[:, 0])):
        raise OrientationError('Cannot drop coordinates when applying orientation to data')
    for ax, flip in enumerate(ornt[:, 1]):
        if flip == -1:
            t_arr = np.flip(t_arr, axis=ax)
    full_transpose = np.arange(t_arr.ndim)
    full_transpose[:n] = np.argsort(ornt[:, 0])
    t_arr = t_arr.transpose(full_transpose)
    return t_arr