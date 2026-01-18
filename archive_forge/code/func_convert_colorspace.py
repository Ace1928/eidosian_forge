from warnings import warn
import numpy as np
from scipy import linalg
from .._shared.utils import (
from ..util import dtype, dtype_limits
def convert_colorspace(arr, fromspace, tospace, *, channel_axis=-1):
    """Convert an image array to a new color space.

    Valid color spaces are:
        'RGB', 'HSV', 'RGB CIE', 'XYZ', 'YUV', 'YIQ', 'YPbPr', 'YCbCr', 'YDbDr'

    Parameters
    ----------
    arr : (..., C=3, ...) array_like
        The image to convert. By default, the final dimension denotes
        channels.
    fromspace : str
        The color space to convert from. Can be specified in lower case.
    tospace : str
        The color space to convert to. Can be specified in lower case.
    channel_axis : int, optional
        This parameter indicates which axis of the array corresponds to
        channels.

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.

    Returns
    -------
    out : (..., C=3, ...) ndarray
        The converted image. Same dimensions as input.

    Raises
    ------
    ValueError
        If fromspace is not a valid color space
    ValueError
        If tospace is not a valid color space

    Notes
    -----
    Conversion is performed through the "central" RGB color space,
    i.e. conversion from XYZ to HSV is implemented as ``XYZ -> RGB -> HSV``
    instead of directly.

    Examples
    --------
    >>> from skimage import data
    >>> img = data.astronaut()
    >>> img_hsv = convert_colorspace(img, 'RGB', 'HSV')
    """
    fromdict = {'rgb': identity, 'hsv': hsv2rgb, 'rgb cie': rgbcie2rgb, 'xyz': xyz2rgb, 'yuv': yuv2rgb, 'yiq': yiq2rgb, 'ypbpr': ypbpr2rgb, 'ycbcr': ycbcr2rgb, 'ydbdr': ydbdr2rgb}
    todict = {'rgb': identity, 'hsv': rgb2hsv, 'rgb cie': rgb2rgbcie, 'xyz': rgb2xyz, 'yuv': rgb2yuv, 'yiq': rgb2yiq, 'ypbpr': rgb2ypbpr, 'ycbcr': rgb2ycbcr, 'ydbdr': rgb2ydbdr}
    fromspace = fromspace.lower()
    tospace = tospace.lower()
    if fromspace not in fromdict:
        msg = f'`fromspace` has to be one of {fromdict.keys()}'
        raise ValueError(msg)
    if tospace not in todict:
        msg = f'`tospace` has to be one of {todict.keys()}'
        raise ValueError(msg)
    return todict[tospace](fromdict[fromspace](arr, channel_axis=channel_axis), channel_axis=channel_axis)