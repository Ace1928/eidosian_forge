import os
import inspect
import warnings
import colorsys
from contextlib import contextmanager
from urllib.request import urlopen, urlretrieve
from types import ModuleType
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.colors import to_rgb
import matplotlib.pyplot as plt
from matplotlib.cbook import normalize_kwargs
from seaborn._core.typing import deprecated
from seaborn.external.version import Version
from seaborn.external.appdirs import user_cache_dir
def desaturate(color, prop):
    """Decrease the saturation channel of a color by some percent.

    Parameters
    ----------
    color : matplotlib color
        hex, rgb-tuple, or html color name
    prop : float
        saturation channel of color will be multiplied by this value

    Returns
    -------
    new_color : rgb tuple
        desaturated color code in RGB tuple representation

    """
    if not 0 <= prop <= 1:
        raise ValueError('prop must be between 0 and 1')
    rgb = to_rgb(color)
    if prop == 1:
        return rgb
    h, l, s = colorsys.rgb_to_hls(*rgb)
    s *= prop
    new_color = colorsys.hls_to_rgb(h, l, s)
    return new_color