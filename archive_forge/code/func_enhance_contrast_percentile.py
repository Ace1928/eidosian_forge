from ..._shared.utils import check_nD
from . import percentile_cy
from .generic import _preprocess_input
def enhance_contrast_percentile(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, p0=0, p1=1):
    """Enhance contrast of an image.

    This replaces each pixel by the local maximum if the pixel grayvalue is
    closer to the local maximum than the local minimum. Otherwise it is
    replaced by the local minimum.

    Only grayvalues between percentiles [p0, p1] are considered in the filter.

    Parameters
    ----------
    image : 2-D array (uint8, uint16)
        Input image.
    footprint : 2-D array
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : 2-D array (same dtype as input)
        If None, a new array is allocated.
    mask : ndarray
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the footprint center point. Shift is bounded to the
        footprint sizes (center must be inside the given footprint).
    p0, p1 : float in [0, ..., 1]
        Define the [p0, p1] percentile interval to be considered for computing
        the value.

    Returns
    -------
    out : 2-D array (same dtype as input image)
        Output image.

    """
    return _apply(percentile_cy._enhance_contrast, image, footprint, out=out, mask=mask, shift_x=shift_x, shift_y=shift_y, p0=p0, p1=p1)