from warnings import warn
import numpy as np
from scipy import linalg
from .._shared.utils import (
from ..util import dtype, dtype_limits
@channel_as_last_axis()
def rgb2lab(rgb, illuminant='D65', observer='2', *, channel_axis=-1):
    """Conversion from the sRGB color space (IEC 61966-2-1:1999)
    to the CIE Lab colorspace under the given illuminant and observer.

    Parameters
    ----------
    rgb : (..., C=3, ...) array_like
        The image in RGB format. By default, the final dimension denotes
        channels.
    illuminant : {"A", "B", "C", "D50", "D55", "D65", "D75", "E"}, optional
        The name of the illuminant (the function is NOT case sensitive).
    observer : {"2", "10", "R"}, optional
        The aperture angle of the observer.
    channel_axis : int, optional
        This parameter indicates which axis of the array corresponds to
        channels.

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.

    Returns
    -------
    out : (..., C=3, ...) ndarray
        The image in Lab format. Same dimensions as input.

    Raises
    ------
    ValueError
        If `rgb` is not at least 2-D with shape (..., C=3, ...).

    Notes
    -----
    RGB is a device-dependent color space so, if you use this function, be
    sure that the image you are analyzing has been mapped to the sRGB color
    space.

    This function uses rgb2xyz and xyz2lab.
    By default Observer="2", Illuminant="D65". CIE XYZ tristimulus values
    x_ref=95.047, y_ref=100., z_ref=108.883. See function
    :func:`~.xyz_tristimulus_values` for a list of supported illuminants.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Standard_illuminant
    """
    return xyz2lab(rgb2xyz(rgb), illuminant, observer)