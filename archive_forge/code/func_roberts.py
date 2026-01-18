import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import binary_erosion, convolve
from .._shared.utils import _supported_float_type, check_nD
from ..restoration.uft import laplacian
from ..util.dtype import img_as_float
def roberts(image, mask=None):
    """Find the edge magnitude using Roberts' cross operator.

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
        The Roberts' Cross edge map.

    See also
    --------
    roberts_pos_diag, roberts_neg_diag : diagonal edge detection.
    sobel, scharr, prewitt, skimage.feature.canny

    Examples
    --------
    >>> from skimage import data
    >>> camera = data.camera()
    >>> from skimage import filters
    >>> edges = filters.roberts(camera)

    """
    check_nD(image, 2)
    out = np.sqrt(roberts_pos_diag(image, mask) ** 2 + roberts_neg_diag(image, mask) ** 2)
    out /= np.sqrt(2)
    return out