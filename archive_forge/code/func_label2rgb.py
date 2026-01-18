import itertools
import numpy as np
from .._shared.utils import _supported_float_type, warn
from ..util import img_as_float
from . import rgb_colors
from .colorconv import gray2rgb, rgb2hsv, hsv2rgb
def label2rgb(label, image=None, colors=None, alpha=0.3, bg_label=0, bg_color=(0, 0, 0), image_alpha=1, kind='overlay', *, saturation=0, channel_axis=-1):
    """Return an RGB image where color-coded labels are painted over the image.

    Parameters
    ----------
    label : ndarray
        Integer array of labels with the same shape as `image`.
    image : ndarray, optional
        Image used as underlay for labels. It should have the same shape as
        `labels`, optionally with an additional RGB (channels) axis. If `image`
        is an RGB image, it is converted to grayscale before coloring.
    colors : list, optional
        List of colors. If the number of labels exceeds the number of colors,
        then the colors are cycled.
    alpha : float [0, 1], optional
        Opacity of colorized labels. Ignored if image is `None`.
    bg_label : int, optional
        Label that's treated as the background. If `bg_label` is specified,
        `bg_color` is `None`, and `kind` is `overlay`,
        background is not painted by any colors.
    bg_color : str or array, optional
        Background color. Must be a name in ``skimage.color.color_dict`` or RGB float
        values between [0, 1].
    image_alpha : float [0, 1], optional
        Opacity of the image.
    kind : string, one of {'overlay', 'avg'}
        The kind of color image desired. 'overlay' cycles over defined colors
        and overlays the colored labels over the original image. 'avg' replaces
        each labeled segment with its average color, for a stained-class or
        pastel painting appearance.
    saturation : float [0, 1], optional
        Parameter to control the saturation applied to the original image
        between fully saturated (original RGB, `saturation=1`) and fully
        unsaturated (grayscale, `saturation=0`). Only applies when
        `kind='overlay'`.
    channel_axis : int, optional
        This parameter indicates which axis of the output array will correspond
        to channels. If `image` is provided, this must also match the axis of
        `image` that corresponds to channels.

        .. versionadded:: 0.19
            ``channel_axis`` was added in 0.19.

    Returns
    -------
    result : ndarray of float, same shape as `image`
        The result of blending a cycling colormap (`colors`) for each distinct
        value in `label` with the image, at a certain alpha value.
    """
    if image is not None:
        image = np.moveaxis(image, source=channel_axis, destination=-1)
    if kind == 'overlay':
        rgb = _label2rgb_overlay(label, image, colors, alpha, bg_label, bg_color, image_alpha, saturation)
    elif kind == 'avg':
        rgb = _label2rgb_avg(label, image, bg_label, bg_color)
    else:
        raise ValueError("`kind` must be either 'overlay' or 'avg'.")
    return np.moveaxis(rgb, source=-1, destination=channel_axis)