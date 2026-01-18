import numpy as np
from scipy import ndimage as ndi
from ..._shared.utils import check_nD, warn
from ...morphology.footprints import _footprint_is_sequence
from ...util import img_as_ubyte
from . import generic_cy
def majority(image, footprint, *, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0):
    """Assign to each pixel the most common value within its neighborhood.

    Parameters
    ----------
    image : ndarray
        Image array (uint8, uint16 array).
    footprint : 2-D array (integer or float)
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray (integer or float), optional
        If None, a new array will be allocated.
    mask : ndarray (integer or float), optional
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int, optional
        Offset added to the footprint center point. Shift is bounded to the
        footprint sizes (center must be inside the given footprint).

    Returns
    -------
    out : 2-D array (same dtype as input image)
        Output image.

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.filters.rank import majority
    >>> from skimage.morphology import disk, ball
    >>> import numpy as np
    >>> img = data.camera()
    >>> rng = np.random.default_rng()
    >>> volume = rng.integers(0, 255, size=(10,10,10), dtype=np.uint8)
    >>> maj_img = majority(img, disk(5))
    >>> maj_img_vol = majority(volume, ball(5))

    """
    np_image = np.asanyarray(image)
    if np_image.ndim == 2:
        return _apply_scalar_per_pixel(generic_cy._majority, image, footprint, out=out, mask=mask, shift_x=shift_x, shift_y=shift_y)
    elif np_image.ndim == 3:
        return _apply_scalar_per_pixel_3D(generic_cy._majority_3D, image, footprint, out=out, mask=mask, shift_x=shift_x, shift_y=shift_y, shift_z=shift_z)
    raise ValueError(f'`image` must have 2 or 3 dimensions, got {np_image.ndim}.')