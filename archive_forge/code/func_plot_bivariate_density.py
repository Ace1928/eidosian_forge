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
def plot_bivariate_density(self, common_norm, fill, levels, thresh, color, legend, cbar, warn_singular, cbar_ax, cbar_kws, estimate_kws, **contour_kws):
    contour_kws = contour_kws.copy()
    estimator = KDE(**estimate_kws)
    if not set(self.variables) - {'x', 'y'}:
        common_norm = False
    all_data = self.plot_data.dropna()
    densities, supports = ({}, {})
    for sub_vars, sub_data in self.iter_data('hue', from_comp_data=True):
        observations = sub_data[['x', 'y']]
        min_variance = observations.var().fillna(0).min()
        observations = (observations['x'], observations['y'])
        if 'weights' in self.variables:
            weights = sub_data['weights']
        else:
            weights = None
        singular = math.isclose(min_variance, 0)
        try:
            if not singular:
                density, support = estimator(*observations, weights=weights)
        except np.linalg.LinAlgError:
            singular = True
        if singular:
            msg = 'KDE cannot be estimated (0 variance or perfect covariance). Pass `warn_singular=False` to disable this warning.'
            if warn_singular:
                warnings.warn(msg, UserWarning, stacklevel=3)
            continue
        ax = self._get_axes(sub_vars)
        _, inv_x = _get_transform_functions(ax, 'x')
        _, inv_y = _get_transform_functions(ax, 'y')
        support = (inv_x(support[0]), inv_y(support[1]))
        if common_norm:
            density *= len(sub_data) / len(all_data)
        key = tuple(sub_vars.items())
        densities[key] = density
        supports[key] = support
    if thresh is None:
        thresh = 0
    if isinstance(levels, Number):
        levels = np.linspace(thresh, 1, levels)
    elif min(levels) < 0 or max(levels) > 1:
        raise ValueError('levels must be in [0, 1]')
    if common_norm:
        common_levels = self._quantile_to_level(list(densities.values()), levels)
        draw_levels = {k: common_levels for k in densities}
    else:
        draw_levels = {k: self._quantile_to_level(d, levels) for k, d in densities.items()}
    if 'hue' in self.variables:
        for param in ['cmap', 'colors']:
            if param in contour_kws:
                msg = f'{param} parameter ignored when using hue mapping.'
                warnings.warn(msg, UserWarning)
                contour_kws.pop(param)
    else:
        coloring_given = set(contour_kws) & {'cmap', 'colors'}
        if fill and (not coloring_given):
            cmap = self._cmap_from_color(color)
            contour_kws['cmap'] = cmap
        if not fill and (not coloring_given):
            contour_kws['colors'] = [color]
        cmap = contour_kws.pop('cmap', None)
        if isinstance(cmap, str):
            cmap = color_palette(cmap, as_cmap=True)
        if cmap is not None:
            contour_kws['cmap'] = cmap
    for sub_vars, _ in self.iter_data('hue'):
        if 'hue' in sub_vars:
            color = self._hue_map(sub_vars['hue'])
            if fill:
                contour_kws['cmap'] = self._cmap_from_color(color)
            else:
                contour_kws['colors'] = [color]
        ax = self._get_axes(sub_vars)
        if fill:
            contour_func = ax.contourf
        else:
            contour_func = ax.contour
        key = tuple(sub_vars.items())
        if key not in densities:
            continue
        density = densities[key]
        xx, yy = supports[key]
        contour_kws.pop('label', None)
        cset = contour_func(xx, yy, density, levels=draw_levels[key], **contour_kws)
        if cbar:
            cbar_kws = {} if cbar_kws is None else cbar_kws
            ax.figure.colorbar(cset, cbar_ax, ax, **cbar_kws)
    ax = self.ax if self.ax is not None else self.facets.axes.flat[0]
    self._add_axis_labels(ax)
    if 'hue' in self.variables and legend:
        artist_kws = {}
        if fill:
            artist = partial(mpl.patches.Patch)
        else:
            artist = partial(mpl.lines.Line2D, [], [])
        ax_obj = self.ax if self.ax is not None else self.facets
        self._add_legend(ax_obj, artist, fill, False, 'layer', 1, artist_kws, {})