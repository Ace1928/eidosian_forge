from warnings import warn
import numpy as np
from scipy import linalg
from .._shared.utils import (
from ..util import dtype, dtype_limits
def rgba2rgb(rgba, background=(1, 1, 1), *, channel_axis=-1):
    """RGBA to RGB conversion using alpha blending [1]_.

    Parameters
    ----------
    rgba : (..., C=4, ...) array_like
        The image in RGBA format. By default, the final dimension denotes
        channels.
    background : array_like
        The color of the background to blend the image with (3 floats
        between 0 to 1 - the RGB value of the background).
    channel_axis : int, optional
        This parameter indicates which axis of the array corresponds to
        channels.

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.

    Returns
    -------
    out : (..., C=3, ...) ndarray
        The image in RGB format. Same dimensions as input.

    Raises
    ------
    ValueError
        If `rgba` is not at least 2D with shape (..., 4, ...).

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Alpha_compositing#Alpha_blending

    Examples
    --------
    >>> from skimage import color
    >>> from skimage import data
    >>> img_rgba = data.logo()
    >>> img_rgb = color.rgba2rgb(img_rgba)
    """
    arr = np.asanyarray(rgba)
    _validate_channel_axis(channel_axis, arr.ndim)
    channel_axis = channel_axis % arr.ndim
    if arr.shape[channel_axis] != 4:
        msg = f'the input array must have size 4 along `channel_axis`, got {arr.shape}'
        raise ValueError(msg)
    float_dtype = _supported_float_type(arr.dtype)
    if float_dtype == np.float32:
        arr = dtype.img_as_float32(arr)
    else:
        arr = dtype.img_as_float64(arr)
    background = np.ravel(background).astype(arr.dtype)
    if len(background) != 3:
        raise ValueError(f'background must be an array-like containing 3 RGB values. Got {len(background)} items')
    if np.any(background < 0) or np.any(background > 1):
        raise ValueError('background RGB values must be floats between 0 and 1.')
    background = reshape_nd(background, arr.ndim, channel_axis)
    alpha = arr[slice_at_axis(slice(3, 4), axis=channel_axis)]
    channels = arr[slice_at_axis(slice(3), axis=channel_axis)]
    out = np.clip((1 - alpha) * background + alpha * channels, a_min=0, a_max=1)
    return out