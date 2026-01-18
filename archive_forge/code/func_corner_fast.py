import functools
import math
from itertools import combinations_with_replacement
import numpy as np
from scipy import ndimage as ndi
from scipy import spatial, stats
from .._shared.filters import gaussian
from .._shared.utils import _supported_float_type, safe_as_int, warn
from ..transform import integral_image
from ..util import img_as_float
from ._hessian_det_appx import _hessian_matrix_det
from .corner_cy import _corner_fast, _corner_moravec, _corner_orientations
from .peak import peak_local_max
from .util import _prepare_grayscale_input_2D, _prepare_grayscale_input_nD
def corner_fast(image, n=12, threshold=0.15):
    """Extract FAST corners for a given image.

    Parameters
    ----------
    image : (M, N) ndarray
        Input image.
    n : int, optional
        Minimum number of consecutive pixels out of 16 pixels on the circle
        that should all be either brighter or darker w.r.t testpixel.
        A point c on the circle is darker w.r.t test pixel p if
        `Ic < Ip - threshold` and brighter if `Ic > Ip + threshold`. Also
        stands for the n in `FAST-n` corner detector.
    threshold : float, optional
        Threshold used in deciding whether the pixels on the circle are
        brighter, darker or similar w.r.t. the test pixel. Decrease the
        threshold when more corners are desired and vice-versa.

    Returns
    -------
    response : ndarray
        FAST corner response image.

    References
    ----------
    .. [1] Rosten, E., & Drummond, T. (2006, May). Machine learning for
           high-speed corner detection. In European conference on computer
           vision (pp. 430-443). Springer, Berlin, Heidelberg.
           :DOI:`10.1007/11744023_34`
           http://www.edwardrosten.com/work/rosten_2006_machine.pdf
    .. [2] Wikipedia, "Features from accelerated segment test",
           https://en.wikipedia.org/wiki/Features_from_accelerated_segment_test

    Examples
    --------
    >>> from skimage.feature import corner_fast, corner_peaks
    >>> square = np.zeros((12, 12))
    >>> square[3:9, 3:9] = 1
    >>> square.astype(int)
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    >>> corner_peaks(corner_fast(square, 9), min_distance=1)
    array([[3, 3],
           [3, 8],
           [8, 3],
           [8, 8]])

    """
    image = _prepare_grayscale_input_2D(image)
    image = np.ascontiguousarray(image)
    response = _corner_fast(image, n, threshold)
    return response