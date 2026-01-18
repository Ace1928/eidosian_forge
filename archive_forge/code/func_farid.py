import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import binary_erosion, convolve
from .._shared.utils import _supported_float_type, check_nD
from ..restoration.uft import laplacian
from ..util.dtype import img_as_float
def farid(image, mask=None, *, axis=None, mode='reflect', cval=0.0):
    """Find the edge magnitude using the Farid transform.

    Parameters
    ----------
    image : array
        The input image.
    mask : array of bool, optional
        Clip the output image to this mask. (Values where mask=0 will be set
        to 0.)
    axis : int or sequence of int, optional
        Compute the edge filter along this axis. If not provided, the edge
        magnitude is computed. This is defined as::

            farid_mag = np.sqrt(sum([farid(image, axis=i)**2
                                     for i in range(image.ndim)]) / image.ndim)

        The magnitude is also computed if axis is a sequence.
    mode : str or sequence of str, optional
        The boundary mode for the convolution. See `scipy.ndimage.convolve`
        for a description of the modes. This can be either a single boundary
        mode or one boundary mode per axis.
    cval : float, optional
        When `mode` is ``'constant'``, this is the constant used in values
        outside the boundary of the image data.

    Returns
    -------
    output : array of float
        The Farid edge map.

    See also
    --------
    farid_h, farid_v : horizontal and vertical edge detection.
    scharr, sobel, prewitt, skimage.feature.canny

    Notes
    -----
    Take the square root of the sum of the squares of the horizontal and
    vertical derivatives to get a magnitude that is somewhat insensitive to
    direction. Similar to the Scharr operator, this operator is designed with
    a rotation invariance constraint.

    References
    ----------
    .. [1] Farid, H. and Simoncelli, E. P., "Differentiation of discrete
           multidimensional signals", IEEE Transactions on Image Processing
           13(4): 496-508, 2004. :DOI:`10.1109/TIP.2004.823819`
    .. [2] Wikipedia, "Farid and Simoncelli Derivatives." Available at:
           <https://en.wikipedia.org/wiki/Image_derivatives#Farid_and_Simoncelli_Derivatives>

    Examples
    --------
    >>> from skimage import data
    >>> camera = data.camera()
    >>> from skimage import filters
    >>> edges = filters.farid(camera)
    """
    output = _generic_edge_filter(image, smooth_weights=farid_smooth, edge_weights=farid_edge, axis=axis, mode=mode, cval=cval)
    output = _mask_filter_result(output, mask)
    return output