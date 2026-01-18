from warnings import warn
import numpy as np
from scipy import linalg
from .._shared.utils import (
from ..util import dtype, dtype_limits
@channel_as_last_axis()
def rgb2ycbcr(rgb, *, channel_axis=-1):
    """RGB to YCbCr color space conversion.

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
        The image in YCbCr format. Same dimensions as input.

    Raises
    ------
    ValueError
        If `rgb` is not at least 2-D with shape (..., C=3, ...).

    Notes
    -----
    Y is between 16 and 235. This is the color space commonly used by video
    codecs; it is sometimes incorrectly called "YUV".

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/YCbCr
    """
    arr = _convert(ycbcr_from_rgb, rgb)
    arr[..., 0] += 16
    arr[..., 1] += 128
    arr[..., 2] += 128
    return arr