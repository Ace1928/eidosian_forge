import colorsys
from itertools import cycle
import numpy as np
import matplotlib as mpl
from .external import husl
from .utils import desaturate, get_color_cycle
from .colors import xkcd_rgb, crayons
from ._compat import get_colormap
def husl_palette(n_colors=6, h=0.01, s=0.9, l=0.65, as_cmap=False):
    """
    Return hues with constant lightness and saturation in the HUSL system.

    The hues are evenly sampled along a circular path. The resulting palette will be
    appropriate for categorical or cyclical data.

    The `h`, `l`, and `s` values should be between 0 and 1.

    This function is similar to :func:`hls_palette`, but it uses a nonlinear color
    space that is more perceptually uniform.

    Parameters
    ----------
    n_colors : int
        Number of colors in the palette.
    h : float
        The value of the first hue.
    l : float
        The lightness value.
    s : float
        The saturation intensity.
    as_cmap : bool
        If True, return a matplotlib colormap object.

    Returns
    -------
    palette
        list of RGB tuples or :class:`matplotlib.colors.ListedColormap`

    See Also
    --------
    hls_palette : Make a palette using evenly spaced hues in the HSL system.

    Examples
    --------
    .. include:: ../docstrings/husl_palette.rst

    """
    if as_cmap:
        n_colors = 256
    hues = np.linspace(0, 1, int(n_colors) + 1)[:-1]
    hues += h
    hues %= 1
    hues *= 359
    s *= 99
    l *= 99
    palette = [_color_to_rgb((h_i, s, l), input='husl') for h_i in hues]
    if as_cmap:
        return mpl.colors.ListedColormap(palette, 'hsl')
    else:
        return _ColorPalette(palette)