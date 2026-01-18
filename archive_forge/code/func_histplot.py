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
def histplot(data=None, *, x=None, y=None, hue=None, weights=None, stat='count', bins='auto', binwidth=None, binrange=None, discrete=None, cumulative=False, common_bins=True, common_norm=True, multiple='layer', element='bars', fill=True, shrink=1, kde=False, kde_kws=None, line_kws=None, thresh=0, pthresh=None, pmax=None, cbar=False, cbar_ax=None, cbar_kws=None, palette=None, hue_order=None, hue_norm=None, color=None, log_scale=None, legend=True, ax=None, **kwargs):
    p = _DistributionPlotter(data=data, variables=dict(x=x, y=y, hue=hue, weights=weights))
    p.map_hue(palette=palette, order=hue_order, norm=hue_norm)
    if ax is None:
        ax = plt.gca()
    p._attach(ax, log_scale=log_scale)
    if p.univariate:
        if fill:
            method = ax.bar if element == 'bars' else ax.fill_between
        else:
            method = ax.plot
        color = _default_color(method, hue, color, kwargs)
    if not p.has_xy_data:
        return ax
    if discrete is None:
        discrete = p._default_discrete()
    estimate_kws = dict(stat=stat, bins=bins, binwidth=binwidth, binrange=binrange, discrete=discrete, cumulative=cumulative)
    if p.univariate:
        p.plot_univariate_histogram(multiple=multiple, element=element, fill=fill, shrink=shrink, common_norm=common_norm, common_bins=common_bins, kde=kde, kde_kws=kde_kws, color=color, legend=legend, estimate_kws=estimate_kws, line_kws=line_kws, **kwargs)
    else:
        p.plot_bivariate_histogram(common_bins=common_bins, common_norm=common_norm, thresh=thresh, pthresh=pthresh, pmax=pmax, color=color, legend=legend, cbar=cbar, cbar_ax=cbar_ax, cbar_kws=cbar_kws, estimate_kws=estimate_kws, **kwargs)
    return ax