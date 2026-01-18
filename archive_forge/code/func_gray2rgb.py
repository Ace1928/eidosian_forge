from warnings import warn
import numpy as np
from scipy import linalg
from .._shared.utils import (
from ..util import dtype, dtype_limits
def gray2rgb(image, *, channel_axis=-1):
    """Create an RGB representation of a gray-level image.

    Parameters
    ----------
    image : array_like
        Input image.
    channel_axis : int, optional
        This parameter indicates which axis of the output array will correspond
        to channels.

    Returns
    -------
    rgb : (..., C=3, ...) ndarray
        RGB image. A new dimension of length 3 is added to input image.

    Notes
    -----
    If the input is a 1-dimensional image of shape ``(M,)``, the output
    will be shape ``(M, C=3)``.
    """
    return np.stack(3 * (image,), axis=channel_axis)