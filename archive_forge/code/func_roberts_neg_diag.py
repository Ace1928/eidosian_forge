import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import binary_erosion, convolve
from .._shared.utils import _supported_float_type, check_nD
from ..restoration.uft import laplacian
from ..util.dtype import img_as_float
def roberts_neg_diag(image, mask=None):
    """Find the cross edges of an image using the Roberts' Cross operator.

    The kernel is applied to the input image to produce separate measurements
    of the gradient component one orientation.

    Parameters
    ----------
    image : 2-D array
        Image to process.
    mask : 2-D array, optional
        An optional mask to limit the application to a certain area.
        Note that pixels surrounding masked regions are also masked to
        prevent masked regions from affecting the result.

    Returns
    -------
    output : 2-D array
        The Robert's edge map.

    Notes
    -----
    We use the following kernel::

      0   1
     -1   0

    """
    check_nD(image, 2)
    if image.dtype.kind == 'f':
        float_dtype = _supported_float_type(image.dtype)
        image = image.astype(float_dtype, copy=False)
    else:
        image = img_as_float(image)
    result = convolve(image, ROBERTS_ND_WEIGHTS)
    return _mask_filter_result(result, mask)