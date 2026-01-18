import warnings
import matplotlib as mpl
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import pandas as pd
from . import cm
from .axisgrid import Grid
from ._compat import get_colormap
from .utils import (
def dim_ratios(self, colors, dendrogram_ratio, colors_ratio):
    """Get the proportions of the figure taken up by each axes."""
    ratios = [dendrogram_ratio]
    if colors is not None:
        if np.ndim(colors) > 2:
            n_colors = len(colors)
        else:
            n_colors = 1
        ratios += [n_colors * colors_ratio]
    ratios.append(1 - sum(ratios))
    return ratios