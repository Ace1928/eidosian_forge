import math
import numbers
import numpy as np
from .._shared.utils import _supported_float_type
from ..color.adapt_rgb import adapt_rgb, hsv_value
from .exposure import rescale_intensity
from ..util import img_as_uint
def map_histogram(hist, min_val, max_val, n_pixels):
    """Calculate the equalized lookup table (mapping).

    It does so by cumulating the input histogram.
    Histogram bins are assumed to be represented by the last array dimension.

    Parameters
    ----------
    hist : ndarray
        Clipped histogram.
    min_val : int
        Minimum value for mapping.
    max_val : int
        Maximum value for mapping.
    n_pixels : int
        Number of pixels in the region.

    Returns
    -------
    out : ndarray
       Mapped intensity LUT.
    """
    out = np.cumsum(hist, axis=-1).astype(float)
    out *= (max_val - min_val) / n_pixels
    out += min_val
    np.clip(out, a_min=None, a_max=max_val, out=out)
    return out.astype(int)