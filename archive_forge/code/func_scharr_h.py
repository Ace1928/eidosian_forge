import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import binary_erosion, convolve
from .._shared.utils import _supported_float_type, check_nD
from ..restoration.uft import laplacian
from ..util.dtype import img_as_float
def scharr_h(image, mask=None):
    """Find the horizontal edges of an image using the Scharr transform.

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
        The Scharr edge map.

    Notes
    -----
    We use the following kernel::

      3   10   3
      0    0   0
     -3  -10  -3

    References
    ----------
    .. [1] D. Kroon, 2009, Short Paper University Twente, Numerical
           Optimization of Kernel Based Image Derivatives.

    """
    check_nD(image, 2)
    return scharr(image, mask=mask, axis=0)