import inspect
import itertools
import math
from collections import OrderedDict
from collections.abc import Iterable
import numpy as np
from scipy import ndimage as ndi
from .._shared.filters import gaussian
from .._shared.utils import _supported_float_type, warn
from .._shared.version_requirements import require
from ..exposure import histogram
from ..filters._multiotsu import (
from ..transform import integral_image
from ..util import dtype_limits
from ._sparse import _correlate_sparse, _validate_window_size
def apply_hysteresis_threshold(image, low, high):
    """Apply hysteresis thresholding to ``image``.

    This algorithm finds regions where ``image`` is greater than ``high``
    OR ``image`` is greater than ``low`` *and* that region is connected to
    a region greater than ``high``.

    Parameters
    ----------
    image : (M[, ...]) ndarray
        Grayscale input image.
    low : float, or array of same shape as ``image``
        Lower threshold.
    high : float, or array of same shape as ``image``
        Higher threshold.

    Returns
    -------
    thresholded : (M[, ...]) array of bool
        Array in which ``True`` indicates the locations where ``image``
        was above the hysteresis threshold.

    Examples
    --------
    >>> image = np.array([1, 2, 3, 2, 1, 2, 1, 3, 2])
    >>> apply_hysteresis_threshold(image, 1.5, 2.5).astype(int)
    array([0, 1, 1, 1, 0, 0, 0, 1, 1])

    References
    ----------
    .. [1] J. Canny. A computational approach to edge detection.
           IEEE Transactions on Pattern Analysis and Machine Intelligence.
           1986; vol. 8, pp.679-698.
           :DOI:`10.1109/TPAMI.1986.4767851`
    """
    low = np.clip(low, a_min=None, a_max=high)
    mask_low = image > low
    mask_high = image > high
    labels_low, num_labels = ndi.label(mask_low)
    sums = ndi.sum(mask_high, labels_low, np.arange(num_labels + 1))
    connected_to_high = sums > 0
    thresholded = connected_to_high[labels_low]
    return thresholded