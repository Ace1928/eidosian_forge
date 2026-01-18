import numpy as np
import functools
from scipy import ndimage as ndi
from .._shared.utils import warn
def remove_small_holes(ar, area_threshold=64, connectivity=1, *, out=None):
    """Remove contiguous holes smaller than the specified size.

    Parameters
    ----------
    ar : ndarray (arbitrary shape, int or bool type)
        The array containing the connected components of interest.
    area_threshold : int, optional (default: 64)
        The maximum area, in pixels, of a contiguous hole that will be filled.
        Replaces `min_size`.
    connectivity : int, {1, 2, ..., ar.ndim}, optional (default: 1)
        The connectivity defining the neighborhood of a pixel.
    out : ndarray
        Array of the same shape as `ar` and bool dtype, into which the
        output is placed. By default, a new array is created.

    Raises
    ------
    TypeError
        If the input array is of an invalid type, such as float or string.
    ValueError
        If the input array contains negative values.

    Returns
    -------
    out : ndarray, same shape and type as input `ar`
        The input array with small holes within connected components removed.

    Examples
    --------
    >>> from skimage import morphology
    >>> a = np.array([[1, 1, 1, 1, 1, 0],
    ...               [1, 1, 1, 0, 1, 0],
    ...               [1, 0, 0, 1, 1, 0],
    ...               [1, 1, 1, 1, 1, 0]], bool)
    >>> b = morphology.remove_small_holes(a, 2)
    >>> b
    array([[ True,  True,  True,  True,  True, False],
           [ True,  True,  True,  True,  True, False],
           [ True, False, False,  True,  True, False],
           [ True,  True,  True,  True,  True, False]])
    >>> c = morphology.remove_small_holes(a, 2, connectivity=2)
    >>> c
    array([[ True,  True,  True,  True,  True, False],
           [ True,  True,  True, False,  True, False],
           [ True, False, False,  True,  True, False],
           [ True,  True,  True,  True,  True, False]])
    >>> d = morphology.remove_small_holes(a, 2, out=a)
    >>> d is a
    True

    Notes
    -----
    If the array type is int, it is assumed that it contains already-labeled
    objects. The labels are not kept in the output image (this function always
    outputs a bool image). It is suggested that labeling is completed after
    using this function.

    """
    _check_dtype_supported(ar)
    if ar.dtype != bool:
        warn('Any labeled images will be returned as a boolean array. Did you mean to use a boolean array?', UserWarning)
    if out is not None:
        if out.dtype != bool:
            raise TypeError('out dtype must be bool')
    else:
        out = ar.astype(bool, copy=True)
    np.logical_not(ar, out=out)
    out = remove_small_objects(out, area_threshold, connectivity, out=out)
    np.logical_not(out, out=out)
    return out