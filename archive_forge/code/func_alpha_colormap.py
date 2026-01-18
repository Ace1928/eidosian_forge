import warnings
from matplotlib.image import _ImageBase
from matplotlib.patches import Patch
from matplotlib.transforms import Bbox, TransformedBbox, BboxTransform
import matplotlib as mpl
import numpy as np
from . import reductions
from . import transfer_functions as tf
from .colors import Sets1to3
from .core import bypixel, Canvas
def alpha_colormap(color, min_alpha=40, max_alpha=255, N=256):
    """
    Generate a transparency-based monochromatic colormap.

    Parameters
    ----------
    color : str or tuple
        Color name, hex code or RGB tuple.
    min_alpha, max_alpha: int
        Values between 0 - 255 representing the range of alpha values to use for
        colormapped pixels that contain data.

    Returns
    -------
    :class:`matplotlib.colors.LinearSegmentedColormap`

    """
    for a in (min_alpha, max_alpha):
        if a < 0 or a > 255:
            raise ValueError('Alpha values must be integers between 0 and 255')
    r, g, b = mpl.colors.to_rgb(color)
    return mpl.colors.LinearSegmentedColormap('_datashader_alpha', {'red': [(0.0, r, r), (1.0, r, r)], 'green': [(0.0, g, g), (1.0, g, g)], 'blue': [(0.0, b, b), (1.0, b, b)], 'alpha': [(0.0, min_alpha / 255, min_alpha / 255), (1.0, max_alpha / 255, max_alpha / 255)]}, N=N)