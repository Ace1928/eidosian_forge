from warnings import warn
import numpy as np
from scipy import linalg
from .._shared.utils import (
from ..util import dtype, dtype_limits
@channel_as_last_axis()
def rgbcie2rgb(rgbcie, *, channel_axis=-1):
    """RGB CIE to RGB color space conversion.

    Parameters
    ----------
    rgbcie : (..., C=3, ...) array_like
        The image in RGB CIE format. By default, the final dimension denotes
        channels.
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
        If `rgbcie` is not at least 2-D with shape (..., C=3, ...).

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/CIE_1931_color_space

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.color import rgb2rgbcie, rgbcie2rgb
    >>> img = data.astronaut()
    >>> img_rgbcie = rgb2rgbcie(img)
    >>> img_rgb = rgbcie2rgb(img_rgbcie)
    """
    return _convert(rgb_from_rgbcie, rgbcie)