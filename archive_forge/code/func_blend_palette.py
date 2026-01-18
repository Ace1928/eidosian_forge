import colorsys
from itertools import cycle
import numpy as np
import matplotlib as mpl
from .external import husl
from .utils import desaturate, get_color_cycle
from .colors import xkcd_rgb, crayons
from ._compat import get_colormap
def blend_palette(colors, n_colors=6, as_cmap=False, input='rgb'):
    """Make a palette that blends between a list of colors.

    Parameters
    ----------
    colors : sequence of colors in various formats interpreted by `input`
        hex code, html color name, or tuple in `input` space.
    n_colors : int, optional
        Number of colors in the palette.
    as_cmap : bool, optional
        If True, return a :class:`matplotlib.colors.ListedColormap`.

    Returns
    -------
    palette
        list of RGB tuples or :class:`matplotlib.colors.ListedColormap`

    Examples
    --------
    .. include: ../docstrings/blend_palette.rst

    """
    colors = [_color_to_rgb(color, input) for color in colors]
    name = 'blend'
    pal = mpl.colors.LinearSegmentedColormap.from_list(name, colors)
    if not as_cmap:
        rgb_array = pal(np.linspace(0, 1, int(n_colors)))[:, :3]
        pal = _ColorPalette(map(tuple, rgb_array))
    return pal