import colorsys
from itertools import cycle
import numpy as np
import matplotlib as mpl
from .external import husl
from .utils import desaturate, get_color_cycle
from .colors import xkcd_rgb, crayons
from ._compat import get_colormap
def cubehelix_palette(n_colors=6, start=0, rot=0.4, gamma=1.0, hue=0.8, light=0.85, dark=0.15, reverse=False, as_cmap=False):
    """Make a sequential palette from the cubehelix system.

    This produces a colormap with linearly-decreasing (or increasing)
    brightness. That means that information will be preserved if printed to
    black and white or viewed by someone who is colorblind.  "cubehelix" is
    also available as a matplotlib-based palette, but this function gives the
    user more control over the look of the palette and has a different set of
    defaults.

    In addition to using this function, it is also possible to generate a
    cubehelix palette generally in seaborn using a string starting with
    `ch:` and containing other parameters (e.g. `"ch:s=.25,r=-.5"`).

    Parameters
    ----------
    n_colors : int
        Number of colors in the palette.
    start : float, 0 <= start <= 3
        The hue value at the start of the helix.
    rot : float
        Rotations around the hue wheel over the range of the palette.
    gamma : float 0 <= gamma
        Nonlinearity to emphasize dark (gamma < 1) or light (gamma > 1) colors.
    hue : float, 0 <= hue <= 1
        Saturation of the colors.
    dark : float 0 <= dark <= 1
        Intensity of the darkest color in the palette.
    light : float 0 <= light <= 1
        Intensity of the lightest color in the palette.
    reverse : bool
        If True, the palette will go from dark to light.
    as_cmap : bool
        If True, return a :class:`matplotlib.colors.ListedColormap`.

    Returns
    -------
    palette
        list of RGB tuples or :class:`matplotlib.colors.ListedColormap`

    See Also
    --------
    choose_cubehelix_palette : Launch an interactive widget to select cubehelix
                               palette parameters.
    dark_palette : Create a sequential palette with dark low values.
    light_palette : Create a sequential palette with bright low values.

    References
    ----------
    Green, D. A. (2011). "A colour scheme for the display of astronomical
    intensity images". Bulletin of the Astromical Society of India, Vol. 39,
    p. 289-295.

    Examples
    --------
    .. include:: ../docstrings/cubehelix_palette.rst

    """

    def get_color_function(p0, p1):

        def color(x):
            xg = x ** gamma
            a = hue * xg * (1 - xg) / 2
            phi = 2 * np.pi * (start / 3 + rot * x)
            return xg + a * (p0 * np.cos(phi) + p1 * np.sin(phi))
        return color
    cdict = {'red': get_color_function(-0.14861, 1.78277), 'green': get_color_function(-0.29227, -0.90649), 'blue': get_color_function(1.97294, 0.0)}
    cmap = mpl.colors.LinearSegmentedColormap('cubehelix', cdict)
    x = np.linspace(light, dark, int(n_colors))
    pal = cmap(x)[:, :3].tolist()
    if reverse:
        pal = pal[::-1]
    if as_cmap:
        x_256 = np.linspace(light, dark, 256)
        if reverse:
            x_256 = x_256[::-1]
        pal_256 = cmap(x_256)
        cmap = mpl.colors.ListedColormap(pal_256, 'seaborn_cubehelix')
        return cmap
    else:
        return _ColorPalette(pal)