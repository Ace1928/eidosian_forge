from numbers import Number
from functools import partial
import math
import textwrap
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as tx
from matplotlib.cbook import normalize_kwargs
from matplotlib.colors import to_rgba
from matplotlib.collections import LineCollection
from ._base import VectorPlotter
from ._statistics import ECDF, Histogram, KDE
from ._stats.counting import Hist
from .axisgrid import (
from .utils import (
from .palettes import color_palette
from .external import husl
from .external.kde import gaussian_kde
from ._docstrings import (
important parameter. Misspecification of the bandwidth can produce a
def plot_rug(self, height, expand_margins, legend, **kws):
    for sub_vars, sub_data in self.iter_data(from_comp_data=True):
        ax = self._get_axes(sub_vars)
        kws.setdefault('linewidth', 1)
        if expand_margins:
            xmarg, ymarg = ax.margins()
            if 'x' in self.variables:
                ymarg += height * 2
            if 'y' in self.variables:
                xmarg += height * 2
            ax.margins(x=xmarg, y=ymarg)
        if 'hue' in self.variables:
            kws.pop('c', None)
            kws.pop('color', None)
        if 'x' in self.variables:
            self._plot_single_rug(sub_data, 'x', height, ax, kws)
        if 'y' in self.variables:
            self._plot_single_rug(sub_data, 'y', height, ax, kws)
        self._add_axis_labels(ax)
        if 'hue' in self.variables and legend:
            legend_artist = partial(mpl.lines.Line2D, [], [])
            self._add_legend(ax, legend_artist, False, False, None, 1, {}, {})