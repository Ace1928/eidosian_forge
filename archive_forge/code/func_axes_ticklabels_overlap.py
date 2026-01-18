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
def axes_ticklabels_overlap(ax):
    """Return booleans for whether the x and y ticklabels on an Axes overlap.

    Parameters
    ----------
    ax : matplotlib Axes

    Returns
    -------
    x_overlap, y_overlap : booleans
        True when the labels on that axis overlap.

    """
    return (axis_ticklabels_overlap(ax.get_xticklabels()), axis_ticklabels_overlap(ax.get_yticklabels()))