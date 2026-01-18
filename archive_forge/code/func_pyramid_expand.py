import math
import numpy as np
from .._shared.filters import gaussian
from .._shared.utils import convert_to_float
from ._warps import resize
def pyramid_expand(image, upscale=2, sigma=None, order=1, mode='reflect', cval=0, preserve_range=False, *, channel_axis=None):
    """Upsample and then smooth image.

    Parameters
    ----------
    image : ndarray
        Input image.
    upscale : float, optional
        Upscale factor.
    sigma : float, optional
        Sigma for Gaussian filter. Default is `2 * upscale / 6.0` which
        corresponds to a filter mask twice the size of the scale factor that
        covers more than 99% of the Gaussian distribution.
    order : int, optional
        Order of splines used in interpolation of upsampling. See
        `skimage.transform.warp` for detail.
    mode : {'reflect', 'constant', 'edge', 'symmetric', 'wrap'}, optional
        The mode parameter determines how the array borders are handled, where
        cval is the value when mode is equal to 'constant'.
    cval : float, optional
        Value to fill past edges of input if mode is 'constant'.
    preserve_range : bool, optional
        Whether to keep the original range of values. Otherwise, the input
        image is converted according to the conventions of `img_as_float`.
        Also see https://scikit-image.org/docs/dev/user_guide/data_types.html
    channel_axis : int or None, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.

    Returns
    -------
    out : array
        Upsampled and smoothed float image.

    References
    ----------
    .. [1] http://persci.mit.edu/pub_pdfs/pyramid83.pdf

    """
    _check_factor(upscale)
    image = convert_to_float(image, preserve_range)
    if channel_axis is not None:
        channel_axis = channel_axis % image.ndim
        out_shape = tuple((math.ceil(upscale * d) if ax != channel_axis else d for ax, d in enumerate(image.shape)))
    else:
        out_shape = tuple((math.ceil(upscale * d) for d in image.shape))
    if sigma is None:
        sigma = 2 * upscale / 6.0
    resized = resize(image, out_shape, order=order, mode=mode, cval=cval, anti_aliasing=False)
    out = _smooth(resized, sigma, mode, cval, channel_axis)
    return out