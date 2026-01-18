import numpy as np
from scipy import ndimage as ndi
from ..._shared.utils import check_nD, warn
from ...morphology.footprints import _footprint_is_sequence
from ...util import img_as_ubyte
from . import generic_cy
def otsu(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0):
    """Local Otsu's threshold value for each pixel.

    Parameters
    ----------
    image : ([P,] M, N) ndarray (uint8, uint16)
        Input image.
    footprint : ndarray
        The neighborhood expressed as an ndarray of 1's and 0's.
    out : ([P,] M, N) array (same dtype as input)
        If None, a new array is allocated.
    mask : ndarray (integer or float), optional
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y, shift_z : int
        Offset added to the footprint center point. Shift is bounded to the
        footprint sizes (center must be inside the given footprint).

    Returns
    -------
    out : ([P,] M, N) ndarray (same dtype as input image)
        Output image.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Otsu's_method

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.filters.rank import otsu
    >>> from skimage.morphology import disk, ball
    >>> import numpy as np
    >>> img = data.camera()
    >>> rng = np.random.default_rng()
    >>> volume = rng.integers(0, 255, size=(10,10,10), dtype=np.uint8)
    >>> local_otsu = otsu(img, disk(5))
    >>> thresh_image = img >= local_otsu
    >>> local_otsu_vol = otsu(volume, ball(5))
    >>> thresh_image_vol = volume >= local_otsu_vol

    """
    np_image = np.asanyarray(image)
    if np_image.ndim == 2:
        return _apply_scalar_per_pixel(generic_cy._otsu, image, footprint, out=out, mask=mask, shift_x=shift_x, shift_y=shift_y)
    elif np_image.ndim == 3:
        return _apply_scalar_per_pixel_3D(generic_cy._otsu_3D, image, footprint, out=out, mask=mask, shift_x=shift_x, shift_y=shift_y, shift_z=shift_z)
    raise ValueError(f'`image` must have 2 or 3 dimensions, got {np_image.ndim}.')