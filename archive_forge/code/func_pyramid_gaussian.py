import math
import numpy as np
from .._shared.filters import gaussian
from .._shared.utils import convert_to_float
from ._warps import resize
def pyramid_gaussian(image, max_layer=-1, downscale=2, sigma=None, order=1, mode='reflect', cval=0, preserve_range=False, *, channel_axis=None):
    """Yield images of the Gaussian pyramid formed by the input image.

    Recursively applies the `pyramid_reduce` function to the image, and yields
    the downscaled images.

    Note that the first image of the pyramid will be the original, unscaled
    image. The total number of images is `max_layer + 1`. In case all layers
    are computed, the last image is either a one-pixel image or the image where
    the reduction does not change its shape.

    Parameters
    ----------
    image : ndarray
        Input image.
    max_layer : int, optional
        Number of layers for the pyramid. 0th layer is the original image.
        Default is -1 which builds all possible layers.
    downscale : float, optional
        Downscale factor.
    sigma : float, optional
        Sigma for Gaussian filter. Default is `2 * downscale / 6.0` which
        corresponds to a filter mask twice the size of the scale factor that
        covers more than 99% of the Gaussian distribution.
    order : int, optional
        Order of splines used in interpolation of downsampling. See
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
    pyramid : generator
        Generator yielding pyramid layers as float images.

    References
    ----------
    .. [1] http://persci.mit.edu/pub_pdfs/pyramid83.pdf

    """
    _check_factor(downscale)
    image = convert_to_float(image, preserve_range)
    layer = 0
    current_shape = image.shape
    prev_layer_image = image
    yield image
    while layer != max_layer:
        layer += 1
        layer_image = pyramid_reduce(prev_layer_image, downscale, sigma, order, mode, cval, channel_axis=channel_axis)
        prev_shape = current_shape
        prev_layer_image = layer_image
        current_shape = layer_image.shape
        if current_shape == prev_shape:
            break
        yield layer_image