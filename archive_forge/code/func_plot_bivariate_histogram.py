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
def plot_bivariate_histogram(self, common_bins, common_norm, thresh, pthresh, pmax, color, legend, cbar, cbar_ax, cbar_kws, estimate_kws, **plot_kws):
    cbar_kws = {} if cbar_kws is None else cbar_kws.copy()
    estimator = Histogram(**estimate_kws)
    if set(self.variables) - {'x', 'y'}:
        all_data = self.comp_data.dropna()
        if common_bins:
            estimator.define_bin_params(all_data['x'], all_data['y'], all_data.get('weights', None))
    else:
        common_norm = False
    full_heights = []
    for _, sub_data in self.iter_data(from_comp_data=True):
        sub_heights, _ = estimator(sub_data['x'], sub_data['y'], sub_data.get('weights', None))
        full_heights.append(sub_heights)
    common_color_norm = not set(self.variables) - {'x', 'y'} or common_norm
    if pthresh is not None and common_color_norm:
        thresh = self._quantile_to_level(full_heights, pthresh)
    plot_kws.setdefault('vmin', 0)
    if common_color_norm:
        if pmax is not None:
            vmax = self._quantile_to_level(full_heights, pmax)
        else:
            vmax = plot_kws.pop('vmax', max(map(np.max, full_heights)))
    else:
        vmax = None
    if color is None:
        color = 'C0'
    for sub_vars, sub_data in self.iter_data('hue', from_comp_data=True):
        if sub_data.empty:
            continue
        heights, (x_edges, y_edges) = estimator(sub_data['x'], sub_data['y'], weights=sub_data.get('weights', None))
        ax = self._get_axes(sub_vars)
        _, inv_x = _get_transform_functions(ax, 'x')
        _, inv_y = _get_transform_functions(ax, 'y')
        x_edges = inv_x(x_edges)
        y_edges = inv_y(y_edges)
        if estimator.stat != 'count' and common_norm:
            heights *= len(sub_data) / len(all_data)
        artist_kws = plot_kws.copy()
        if 'hue' in self.variables:
            color = self._hue_map(sub_vars['hue'])
            cmap = self._cmap_from_color(color)
            artist_kws['cmap'] = cmap
        else:
            cmap = artist_kws.pop('cmap', None)
            if isinstance(cmap, str):
                cmap = color_palette(cmap, as_cmap=True)
            elif cmap is None:
                cmap = self._cmap_from_color(color)
            artist_kws['cmap'] = cmap
        if not common_color_norm and pmax is not None:
            vmax = self._quantile_to_level(heights, pmax)
        if vmax is not None:
            artist_kws['vmax'] = vmax
        if not common_color_norm and pthresh:
            thresh = self._quantile_to_level(heights, pthresh)
        if thresh is not None:
            heights = np.ma.masked_less_equal(heights, thresh)
        x_grid = any([l.get_visible() for l in ax.xaxis.get_gridlines()])
        y_grid = any([l.get_visible() for l in ax.yaxis.get_gridlines()])
        mesh = ax.pcolormesh(x_edges, y_edges, heights.T, **artist_kws)
        if thresh is not None:
            mesh.sticky_edges.x[:] = []
            mesh.sticky_edges.y[:] = []
        if cbar:
            ax.figure.colorbar(mesh, cbar_ax, ax, **cbar_kws)
        if x_grid:
            ax.grid(True, axis='x')
        if y_grid:
            ax.grid(True, axis='y')
    ax = self.ax if self.ax is not None else self.facets.axes.flat[0]
    self._add_axis_labels(ax)
    if 'hue' in self.variables and legend:
        artist_kws = {}
        artist = partial(mpl.patches.Patch)
        ax_obj = self.ax if self.ax is not None else self.facets
        self._add_legend(ax_obj, artist, True, False, 'layer', 1, artist_kws, {})