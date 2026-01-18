import numpy as np
from .._shared.utils import _supported_float_type
from .colorconv import lab2lch, _cart2polar_2pi
def deltaE_cie76(lab1, lab2, channel_axis=-1):
    """Euclidean distance between two points in Lab color space

    Parameters
    ----------
    lab1 : array_like
        reference color (Lab colorspace)
    lab2 : array_like
        comparison color (Lab colorspace)
    channel_axis : int, optional
        This parameter indicates which axis of the arrays corresponds to
        channels.

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.

    Returns
    -------
    dE : array_like
        distance between colors `lab1` and `lab2`

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Color_difference
    .. [2] A. R. Robertson, "The CIE 1976 color-difference formulae,"
           Color Res. Appl. 2, 7-11 (1977).
    """
    lab1, lab2 = _float_inputs(lab1, lab2, allow_float32=True)
    L1, a1, b1 = np.moveaxis(lab1, source=channel_axis, destination=0)[:3]
    L2, a2, b2 = np.moveaxis(lab2, source=channel_axis, destination=0)[:3]
    return np.sqrt((L2 - L1) ** 2 + (a2 - a1) ** 2 + (b2 - b1) ** 2)