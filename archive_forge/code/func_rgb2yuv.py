from warnings import warn
import numpy as np
from scipy import linalg
from .._shared.utils import (
from ..util import dtype, dtype_limits
@channel_as_last_axis()
def rgb2yuv(rgb, *, channel_axis=-1):
    """RGB to YUV color space conversion.

    Parameters
    ----------
    rgb : (..., C=3, ...) array_like
        The image in RGB format. By default, the final dimension denotes
        channels.
    channel_axis : int, optional
        This parameter indicates which axis of the array corresponds to
        channels.

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.

    Returns
    -------
    out : (..., C=3, ...) ndarray
        The image in YUV format. Same dimensions as input.

    Raises
    ------
    ValueError
        If `rgb` is not at least 2-D with shape (..., C=3, ...).

    Notes
    -----
    Y is between 0 and 1.  Use YCbCr instead of YUV for the color space
    commonly used by video codecs, where Y ranges from 16 to 235.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/YUV
    """
    return _convert(yuv_from_rgb, rgb)