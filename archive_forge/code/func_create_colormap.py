from .. import utils
from .._lazyload import matplotlib as mpl
from .utils import _get_figure
from .utils import parse_fontsize
from .utils import temp_fontsize
import numpy as np
import warnings
@utils._with_pkg(pkg='matplotlib', min_version=3)
def create_colormap(colors, name='scprep_custom_cmap'):
    """Create a custom colormap from a list of colors.

    Parameters
    ----------
    colors : list-like
        List of `matplotlib` colors. Includes RGB, RGBA,
        string color names and more.
        See <https://matplotlib.org/api/colors_api.html>

    Returns
    -------
    cmap : `matplotlib.colors.LinearSegmentedColormap`
        Custom colormap
    """
    if len(colors) == 1:
        colors = np.repeat(colors, 2)
    vals = np.linspace(0, 1, len(colors))
    cdict = dict(red=[], green=[], blue=[], alpha=[])
    for val, color in zip(vals, colors):
        r, g, b, a = mpl.colors.to_rgba(color)
        cdict['red'].append((val, r, r))
        cdict['green'].append((val, g, g))
        cdict['blue'].append((val, b, b))
        cdict['alpha'].append((val, a, a))
    cmap = mpl.colors.LinearSegmentedColormap(name, cdict)
    return cmap