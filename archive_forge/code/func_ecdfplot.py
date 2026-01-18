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
def ecdfplot(data=None, *, x=None, y=None, hue=None, weights=None, stat='proportion', complementary=False, palette=None, hue_order=None, hue_norm=None, log_scale=None, legend=True, ax=None, **kwargs):
    p = _DistributionPlotter(data=data, variables=dict(x=x, y=y, hue=hue, weights=weights))
    p.map_hue(palette=palette, order=hue_order, norm=hue_norm)
    if ax is None:
        ax = plt.gca()
    p._attach(ax, log_scale=log_scale)
    color = kwargs.pop('color', kwargs.pop('c', None))
    kwargs['color'] = _default_color(ax.plot, hue, color, kwargs)
    if not p.has_xy_data:
        return ax
    if not p.univariate:
        raise NotImplementedError('Bivariate ECDF plots are not implemented')
    estimate_kws = dict(stat=stat, complementary=complementary)
    p.plot_univariate_ecdf(estimate_kws=estimate_kws, legend=legend, **kwargs)
    return ax