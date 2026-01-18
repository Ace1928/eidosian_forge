import numpy as np
from .._shared.utils import warn
from ..util import dtype_limits, invert, crop
from . import grayreconstruct, _util
from ._extrema_cy import _local_maxima
def h_minima(image, h, footprint=None):
    """Determine all minima of the image with depth >= h.

    The local minima are defined as connected sets of pixels with equal
    gray level strictly smaller than the gray levels of all pixels in direct
    neighborhood of the set.

    A local minimum M of depth h is a local minimum for which
    there is at least one path joining M with an equal or lower local minimum
    on which the maximal value is f(M) + h (i.e. the values along the path
    are not increasing by more than h with respect to the minimum's value)
    and no path to an equal or lower local minimum for which the maximal
    value is smaller.

    The global minima of the image are also found by this function.

    Parameters
    ----------
    image : ndarray
        The input image for which the minima are to be calculated.
    h : unsigned integer
        The minimal depth of all extracted minima.
    footprint : ndarray, optional
        The neighborhood expressed as an n-D array of 1's and 0's.
        Default is the ball of radius 1 according to the maximum norm
        (i.e. a 3x3 square for 2D images, a 3x3x3 cube for 3D images, etc.)

    Returns
    -------
    h_min : ndarray
        The local minima of depth >= h and the global minima.
        The resulting image is a binary image, where pixels belonging to
        the determined minima take value 1, the others take value 0.

    See Also
    --------
    skimage.morphology.h_maxima
    skimage.morphology.local_maxima
    skimage.morphology.local_minima

    References
    ----------
    .. [1] Soille, P., "Morphological Image Analysis: Principles and
           Applications" (Chapter 6), 2nd edition (2003), ISBN 3540429883.

    Examples
    --------
    >>> import numpy as np
    >>> from skimage.morphology import extrema

    We create an image (quadratic function with a minimum in the center and
    4 additional constant maxima.
    The depth of the minima are: 1, 21, 41, 61, 81

    >>> w = 10
    >>> x, y = np.mgrid[0:w,0:w]
    >>> f = 180 + 0.2*((x - w/2)**2 + (y-w/2)**2)
    >>> f[2:4,2:4] = 160; f[2:4,7:9] = 140; f[7:9,2:4] = 120; f[7:9,7:9] = 100
    >>> f = f.astype(int)

    We can calculate all minima with a depth of at least 40:

    >>> minima = extrema.h_minima(f, 40)

    The resulting image will contain 3 local minima.
    """
    if h > np.ptp(image):
        return np.zeros(image.shape, dtype=np.uint8)
    if np.issubdtype(type(h), np.floating) and np.issubdtype(image.dtype, np.integer):
        if h % 1 != 0:
            warn('possible precision loss converting image to floating point. To silence this warning, ensure image and h have same data type.', stacklevel=2)
            image = image.astype(float)
        else:
            h = image.dtype.type(h)
    if h == 0:
        raise ValueError('h = 0 is ambiguous, use local_minima() instead?')
    if np.issubdtype(image.dtype, np.floating):
        resolution = 2 * np.finfo(image.dtype).resolution * np.abs(image)
        shifted_img = image + h + resolution
    else:
        shifted_img = _add_constant_clip(image, h)
    rec_img = grayreconstruct.reconstruction(shifted_img, image, method='erosion', footprint=footprint)
    residue_img = rec_img - image
    return (residue_img >= h).astype(np.uint8)