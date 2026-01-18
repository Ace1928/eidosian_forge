import warnings
import numpy as np
from scipy import ndimage as ndi
from .footprints import _footprint_is_sequence, mirror_footprint, pad_footprint
from .misc import default_footprint
from .._shared.utils import DEPRECATED
@default_footprint
def dilation(image, footprint=None, out=None, shift_x=DEPRECATED, shift_y=DEPRECATED, *, mode='reflect', cval=0.0):
    """Return grayscale morphological dilation of an image.

    Morphological dilation sets the value of a pixel to the maximum over all
    pixel values within a local neighborhood centered about it. The values
    where the footprint is 1 define this neighborhood.
    Dilation enlarges bright regions and shrinks dark regions.

    Parameters
    ----------
    image : ndarray
        Image array.
    footprint : ndarray or tuple, optional
        The neighborhood expressed as a 2-D array of 1's and 0's.
        If None, use a cross-shaped footprint (connectivity=1). The footprint
        can also be provided as a sequence of smaller footprints as described
        in the notes below.
    out : ndarray, optional
        The array to store the result of the morphology. If None is
        passed, a new array will be allocated.
    mode : str, optional
        The `mode` parameter determines how the array borders are handled.
        Valid modes are: 'reflect', 'constant', 'nearest', 'mirror', 'wrap',
        'max', 'min', or 'ignore'.
        If 'min' or 'ignore', pixels outside the image domain are assumed
        to be the maximum for the image's dtype, which causes them to not
        influence the result. Default is 'reflect'.
    cval : scalar, optional
        Value to fill past edges of input if `mode` is 'constant'. Default
        is 0.0.

        .. versionadded:: 0.23
            `mode` and `cval` were added in 0.23.

    Returns
    -------
    dilated : uint8 array, same shape and type as `image`
        The result of the morphological dilation.

    Other Parameters
    ----------------
    shift_x, shift_y : DEPRECATED

        .. deprecated:: 0.23

    Notes
    -----
    For ``uint8`` (and ``uint16`` up to a certain bit-depth) data, the lower
    algorithm complexity makes the :func:`skimage.filters.rank.maximum`
    function more efficient for larger images and footprints.

    The footprint can also be a provided as a sequence of 2-tuples where the
    first element of each 2-tuple is a footprint ndarray and the second element
    is an integer describing the number of times it should be iterated. For
    example ``footprint=[(np.ones((9, 1)), 1), (np.ones((1, 9)), 1)]``
    would apply a 9x1 footprint followed by a 1x9 footprint resulting in a net
    effect that is the same as ``footprint=np.ones((9, 9))``, but with lower
    computational cost. Most of the builtin footprints such as
    :func:`skimage.morphology.disk` provide an option to automatically generate
    a footprint sequence of this type.

    For non-symmetric footprints, :func:`skimage.morphology.binary_dilation`
    and :func:`skimage.morphology.dilation` produce an output that differs:
    `binary_dilation` mirrors the footprint, whereas `dilation` does not.

    Examples
    --------
    >>> # Dilation enlarges bright regions
    >>> import numpy as np
    >>> from skimage.morphology import square
    >>> bright_pixel = np.array([[0, 0, 0, 0, 0],
    ...                          [0, 0, 0, 0, 0],
    ...                          [0, 0, 1, 0, 0],
    ...                          [0, 0, 0, 0, 0],
    ...                          [0, 0, 0, 0, 0]], dtype=np.uint8)
    >>> dilation(bright_pixel, square(3))
    array([[0, 0, 0, 0, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 0, 0, 0, 0]], dtype=uint8)

    """
    if out is None:
        out = np.empty_like(image)
    if mode not in _SUPPORTED_MODES:
        raise ValueError(f'unsupported mode, got {mode!r}')
    if mode == 'ignore':
        mode = 'min'
    mode, cval = _min_max_to_constant_mode(image.dtype, mode, cval)
    footprint = _shift_footprints(footprint, shift_x, shift_y)
    footprint = pad_footprint(footprint, pad_end=False)
    footprint = mirror_footprint(footprint)
    if not _footprint_is_sequence(footprint):
        footprint = [(footprint, 1)]
    out = _iterate_gray_func(gray_func=ndi.grey_dilation, image=image, footprints=footprint, out=out, mode=mode, cval=cval)
    return out