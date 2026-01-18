import functools
from math import ceil
import numbers
import scipy.stats
import numpy as np
from ..util.dtype import img_as_float
from .._shared import utils
from .._shared.utils import _supported_float_type, warn
from ._denoise_cy import _denoise_bilateral, _denoise_tv_bregman
from .. import color
from ..color.colorconv import ycbcr_from_rgb
@utils.channel_as_last_axis()
def denoise_tv_bregman(image, weight=5.0, max_num_iter=100, eps=0.001, isotropic=True, *, channel_axis=None):
    """Perform total variation denoising using split-Bregman optimization.

    Given :math:`f`, a noisy image (input data),
    total variation denoising (also known as total variation regularization)
    aims to find an image :math:`u` with less total variation than :math:`f`,
    under the constraint that :math:`u` remain similar to :math:`f`.
    This can be expressed by the Rudin--Osher--Fatemi (ROF) minimization
    problem:

    .. math::

        \\min_{u} \\sum_{i=0}^{N-1} \\left( \\left| \\nabla{u_i} \\right| + \\frac{\\lambda}{2}(f_i - u_i)^2 \\right)

    where :math:`\\lambda` is a positive parameter.
    The first term of this cost function is the total variation;
    the second term represents data fidelity. As :math:`\\lambda \\to 0`,
    the total variation term dominates, forcing the solution to have smaller
    total variation, at the expense of looking less like the input data.

    This code is an implementation of the split Bregman algorithm of Goldstein
    and Osher to solve the ROF problem ([1]_, [2]_, [3]_).

    Parameters
    ----------
    image : ndarray
        Input image to be denoised (converted using :func:`~.img_as_float`).
    weight : float, optional
        Denoising weight. It is equal to :math:`\\frac{\\lambda}{2}`. Therefore,
        the smaller the `weight`, the more denoising (at
        the expense of less similarity to `image`).
    eps : float, optional
        Tolerance :math:`\\varepsilon > 0` for the stop criterion:
        The algorithm stops when :math:`\\|u_n - u_{n-1}\\|_2 < \\varepsilon`.
    max_num_iter : int, optional
        Maximal number of iterations used for the optimization.
    isotropic : boolean, optional
        Switch between isotropic and anisotropic TV denoising.
    channel_axis : int or None, optional
        If ``None``, the image is assumed to be grayscale (single-channel).
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.

    Returns
    -------
    u : ndarray
        Denoised image.

    Notes
    -----
    Ensure that `channel_axis` parameter is set appropriately for color
    images.

    The principle of total variation denoising is explained in [4]_.
    It is about minimizing the total variation of an image,
    which can be roughly described as
    the integral of the norm of the image gradient. Total variation
    denoising tends to produce cartoon-like images, that is,
    piecewise-constant images.

    See Also
    --------
    denoise_tv_chambolle : Perform total variation denoising in nD.

    References
    ----------
    .. [1] Tom Goldstein and Stanley Osher, "The Split Bregman Method For L1
           Regularized Problems",
           https://ww3.math.ucla.edu/camreport/cam08-29.pdf
    .. [2] Pascal Getreuer, "Rudin–Osher–Fatemi Total Variation Denoising
           using Split Bregman" in Image Processing On Line on 2012–05–19,
           https://www.ipol.im/pub/art/2012/g-tvd/article_lr.pdf
    .. [3] https://web.math.ucsb.edu/~cgarcia/UGProjects/BregmanAlgorithms_JacquelineBush.pdf
    .. [4] https://en.wikipedia.org/wiki/Total_variation_denoising

    """
    image = np.atleast_3d(img_as_float(image))
    rows = image.shape[0]
    cols = image.shape[1]
    dims = image.shape[2]
    shape_ext = (rows + 2, cols + 2, dims)
    out = np.zeros(shape_ext, image.dtype)
    if channel_axis is not None:
        channel_out = np.zeros(shape_ext[:2] + (1,), dtype=out.dtype)
        for c in range(image.shape[-1]):
            channel_in = np.ascontiguousarray(image[..., c:c + 1])
            _denoise_tv_bregman(channel_in, image.dtype.type(weight), max_num_iter, eps, isotropic, channel_out)
            out[..., c] = channel_out[..., 0]
    else:
        image = np.ascontiguousarray(image)
        _denoise_tv_bregman(image, image.dtype.type(weight), max_num_iter, eps, isotropic, out)
    return np.squeeze(out[1:-1, 1:-1])