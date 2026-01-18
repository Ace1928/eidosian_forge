import functools
import math
from itertools import combinations_with_replacement
import numpy as np
from scipy import ndimage as ndi
from scipy import spatial, stats
from .._shared.filters import gaussian
from .._shared.utils import _supported_float_type, safe_as_int, warn
from ..transform import integral_image
from ..util import img_as_float
from ._hessian_det_appx import _hessian_matrix_det
from .corner_cy import _corner_fast, _corner_moravec, _corner_orientations
from .peak import peak_local_max
from .util import _prepare_grayscale_input_2D, _prepare_grayscale_input_nD
def corner_kitchen_rosenfeld(image, mode='constant', cval=0):
    """Compute Kitchen and Rosenfeld corner measure response image.

    The corner measure is calculated as follows::

        (imxx * imy**2 + imyy * imx**2 - 2 * imxy * imx * imy)
            / (imx**2 + imy**2)

    Where imx and imy are the first and imxx, imxy, imyy the second
    derivatives.

    Parameters
    ----------
    image : (M, N) ndarray
        Input image.
    mode : {'constant', 'reflect', 'wrap', 'nearest', 'mirror'}, optional
        How to handle values outside the image borders.
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.

    Returns
    -------
    response : ndarray
        Kitchen and Rosenfeld response image.

    References
    ----------
    .. [1] Kitchen, L., & Rosenfeld, A. (1982). Gray-level corner detection.
           Pattern recognition letters, 1(2), 95-102.
           :DOI:`10.1016/0167-8655(82)90020-4`
    """
    float_dtype = _supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)
    imy, imx = _compute_derivatives(image, mode=mode, cval=cval)
    imxy, imxx = _compute_derivatives(imx, mode=mode, cval=cval)
    imyy, imyx = _compute_derivatives(imy, mode=mode, cval=cval)
    numerator = imxx * imy ** 2 + imyy * imx ** 2 - 2 * imxy * imx * imy
    denominator = imx ** 2 + imy ** 2
    response = np.zeros_like(image, dtype=float_dtype)
    mask = denominator != 0
    response[mask] = numerator[mask] / denominator[mask]
    return response