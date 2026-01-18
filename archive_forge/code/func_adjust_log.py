import numpy as np
from ..util.dtype import dtype_range, dtype_limits
from .._shared import utils
def adjust_log(image, gain=1, inv=False):
    """Performs Logarithmic correction on the input image.

    This function transforms the input image pixelwise according to the
    equation ``O = gain*log(1 + I)`` after scaling each pixel to the range
    0 to 1. For inverse logarithmic correction, the equation is
    ``O = gain*(2**I - 1)``.

    Parameters
    ----------
    image : ndarray
        Input image.
    gain : float, optional
        The constant multiplier. Default value is 1.
    inv : float, optional
        If True, it performs inverse logarithmic correction,
        else correction will be logarithmic. Defaults to False.

    Returns
    -------
    out : ndarray
        Logarithm corrected output image.

    See Also
    --------
    adjust_gamma

    References
    ----------
    .. [1] http://www.ece.ucsb.edu/Faculty/Manjunath/courses/ece178W03/EnhancePart1.pdf

    """
    _assert_non_negative(image)
    dtype = image.dtype.type
    scale = float(dtype_limits(image, True)[1] - dtype_limits(image, True)[0])
    if inv:
        out = (2 ** (image / scale) - 1) * scale * gain
        return dtype(out)
    out = np.log2(1 + image / scale) * scale * gain
    return out.astype(dtype)