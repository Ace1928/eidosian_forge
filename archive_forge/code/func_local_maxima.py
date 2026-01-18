import numpy as np
from .._shared.utils import warn
from ..util import dtype_limits, invert, crop
from . import grayreconstruct, _util
from ._extrema_cy import _local_maxima
def local_maxima(image, footprint=None, connectivity=None, indices=False, allow_borders=True):
    """Find local maxima of n-dimensional array.

    The local maxima are defined as connected sets of pixels with equal gray
    level (plateaus) strictly greater than the gray levels of all pixels in the
    neighborhood.

    Parameters
    ----------
    image : ndarray
        An n-dimensional array.
    footprint : ndarray, optional
        The footprint (structuring element) used to determine the neighborhood
        of each evaluated pixel (``True`` denotes a connected pixel). It must
        be a boolean array and have the same number of dimensions as `image`.
        If neither `footprint` nor `connectivity` are given, all adjacent
        pixels are considered as part of the neighborhood.
    connectivity : int, optional
        A number used to determine the neighborhood of each evaluated pixel.
        Adjacent pixels whose squared distance from the center is less than or
        equal to `connectivity` are considered neighbors. Ignored if
        `footprint` is not None.
    indices : bool, optional
        If True, the output will be a tuple of one-dimensional arrays
        representing the indices of local maxima in each dimension. If False,
        the output will be a boolean array with the same shape as `image`.
    allow_borders : bool, optional
        If true, plateaus that touch the image border are valid maxima.

    Returns
    -------
    maxima : ndarray or tuple[ndarray]
        If `indices` is false, a boolean array with the same shape as `image`
        is returned with ``True`` indicating the position of local maxima
        (``False`` otherwise). If `indices` is true, a tuple of one-dimensional
        arrays containing the coordinates (indices) of all found maxima.

    Warns
    -----
    UserWarning
        If `allow_borders` is false and any dimension of the given `image` is
        shorter than 3 samples, maxima can't exist and a warning is shown.

    See Also
    --------
    skimage.morphology.local_minima
    skimage.morphology.h_maxima
    skimage.morphology.h_minima

    Notes
    -----
    This function operates on the following ideas:

    1. Make a first pass over the image's last dimension and flag candidates
       for local maxima by comparing pixels in only one direction.
       If the pixels aren't connected in the last dimension all pixels are
       flagged as candidates instead.

    For each candidate:

    2. Perform a flood-fill to find all connected pixels that have the same
       gray value and are part of the plateau.
    3. Consider the connected neighborhood of a plateau: if no bordering sample
       has a higher gray level, mark the plateau as a definite local maximum.

    Examples
    --------
    >>> from skimage.morphology import local_maxima
    >>> image = np.zeros((4, 7), dtype=int)
    >>> image[1:3, 1:3] = 1
    >>> image[3, 0] = 1
    >>> image[1:3, 4:6] = 2
    >>> image[3, 6] = 3
    >>> image
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 0, 2, 2, 0],
           [0, 1, 1, 0, 2, 2, 0],
           [1, 0, 0, 0, 0, 0, 3]])

    Find local maxima by comparing to all neighboring pixels (maximal
    connectivity):

    >>> local_maxima(image)
    array([[False, False, False, False, False, False, False],
           [False,  True,  True, False, False, False, False],
           [False,  True,  True, False, False, False, False],
           [ True, False, False, False, False, False,  True]])
    >>> local_maxima(image, indices=True)
    (array([1, 1, 2, 2, 3, 3]), array([1, 2, 1, 2, 0, 6]))

    Find local maxima without comparing to diagonal pixels (connectivity 1):

    >>> local_maxima(image, connectivity=1)
    array([[False, False, False, False, False, False, False],
           [False,  True,  True, False,  True,  True, False],
           [False,  True,  True, False,  True,  True, False],
           [ True, False, False, False, False, False,  True]])

    and exclude maxima that border the image edge:

    >>> local_maxima(image, connectivity=1, allow_borders=False)
    array([[False, False, False, False, False, False, False],
           [False,  True,  True, False,  True,  True, False],
           [False,  True,  True, False,  True,  True, False],
           [False, False, False, False, False, False, False]])
    """
    image = np.asarray(image, order='C')
    if image.size == 0:
        if indices:
            return np.nonzero(image)
        else:
            return np.zeros(image.shape, dtype=bool)
    if allow_borders:
        image = np.pad(image, 1, mode='constant', constant_values=image.min())
    flags = np.zeros(image.shape, dtype=np.uint8)
    _util._set_border_values(flags, value=3)
    if any((s < 3 for s in image.shape)):
        warn("maxima can't exist for an image with any dimension smaller 3 if borders aren't allowed", stacklevel=3)
    else:
        footprint = _util._resolve_neighborhood(footprint, connectivity, image.ndim)
        neighbor_offsets = _util._offsets_to_raveled_neighbors(image.shape, footprint, center=(1,) * image.ndim)
        try:
            _local_maxima(image.ravel(), flags.ravel(), neighbor_offsets)
        except TypeError:
            if image.dtype == np.float16:
                raise TypeError('dtype of `image` is float16 which is not supported, try upcasting to float32')
            else:
                raise
    if allow_borders:
        flags = crop(flags, 1)
    else:
        _util._set_border_values(flags, value=0)
    if indices:
        return np.nonzero(flags)
    else:
        return flags.view(bool)