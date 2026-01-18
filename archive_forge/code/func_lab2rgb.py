from warnings import warn
import numpy as np
from scipy import linalg
from .._shared.utils import (
from ..util import dtype, dtype_limits
@channel_as_last_axis()
def lab2rgb(lab, illuminant='D65', observer='2', *, channel_axis=-1):
    """Convert image in CIE-LAB to sRGB color space.

    Parameters
    ----------
    lab : (..., C=3, ...) array_like
        The input image in CIE-LAB color space.
        Unless `channel_axis` is set, the final dimension denotes the CIE-LAB
        channels.
        The L* values range from 0 to 100;
        the a* and b* values range from -128 to 127.
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
        The image in sRGB color space, of same shape as input.

    Raises
    ------
    ValueError
        If `lab` is not at least 2-D with shape (..., C=3, ...).

    Notes
    -----
    This function uses :func:`~.lab2xyz` and :func:`~.xyz2rgb`.
    The CIE XYZ tristimulus values are x_ref = 95.047, y_ref = 100., and
    z_ref = 108.883. See function :func:`~.xyz_tristimulus_values` for a list of
    supported illuminants.

    See Also
    --------
    rgb2lab

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Standard_illuminant
    .. [2] https://en.wikipedia.org/wiki/CIELAB_color_space
    """
    xyz, n_invalid = _lab2xyz(lab, illuminant, observer)
    if n_invalid != 0:
        warn(f'Conversion from CIE-LAB, via XYZ to sRGB color space resulted in {n_invalid} negative Z values that have been clipped to zero', stacklevel=3)
    return xyz2rgb(xyz)