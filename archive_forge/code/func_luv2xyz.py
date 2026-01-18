from warnings import warn
import numpy as np
from scipy import linalg
from .._shared.utils import (
from ..util import dtype, dtype_limits
@channel_as_last_axis()
def luv2xyz(luv, illuminant='D65', observer='2', *, channel_axis=-1):
    """CIE-Luv to XYZ color space conversion.

    Parameters
    ----------
    luv : (..., C=3, ...) array_like
        The image in CIE-Luv format. By default, the final dimension denotes
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
        The image in XYZ format. Same dimensions as input.

    Raises
    ------
    ValueError
        If `luv` is not at least 2-D with shape (..., C=3, ...).
    ValueError
        If either the illuminant or the observer angle are not supported or
        unknown.

    Notes
    -----
    XYZ conversion weights use observer=2A. Reference whitepoint for D65
    Illuminant, with XYZ tristimulus values of ``(95.047, 100., 108.883)``. See
    function :func:`~.xyz_tristimulus_values` for a list of supported illuminants.

    References
    ----------
    .. [1] http://www.easyrgb.com/en/math.php
    .. [2] https://en.wikipedia.org/wiki/CIELUV
    """
    arr = _prepare_colorarray(luv, channel_axis=-1).copy()
    L, u, v = (arr[..., 0], arr[..., 1], arr[..., 2])
    eps = np.finfo(arr.dtype).eps
    y = L.copy()
    mask = y > 7.999625
    y[mask] = np.power((y[mask] + 16.0) / 116.0, 3.0)
    y[~mask] = y[~mask] / 903.3
    xyz_ref_white = xyz_tristimulus_values(illuminant=illuminant, observer=observer, dtype=arr.dtype)
    y *= xyz_ref_white[1]
    uv_weights = np.array([1, 15, 3], dtype=arr.dtype)
    u0 = 4 * xyz_ref_white[0] / (uv_weights @ xyz_ref_white)
    v0 = 9 * xyz_ref_white[1] / (uv_weights @ xyz_ref_white)
    a = u0 + u / (13.0 * L + eps)
    b = v0 + v / (13.0 * L + eps)
    c = 3 * y * (5 * b - 3)
    z = ((a - 4) * c - 15 * a * b * y) / (12 * b)
    x = -(c / b + 3.0 * z)
    return np.concatenate([q[..., np.newaxis] for q in [x, y, z]], axis=-1)