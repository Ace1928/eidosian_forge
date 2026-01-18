from collections import namedtuple
from textwrap import dedent
import warnings
from colorsys import rgb_to_hls
from functools import partial
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.cbook import normalize_kwargs
from matplotlib.collections import PatchCollection
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from seaborn._core.typing import default, deprecated
from seaborn._base import VectorPlotter, infer_orient, categorical_order
from seaborn._stats.density import KDE
from seaborn import utils
from seaborn.utils import (
from seaborn._compat import groupby_apply_include_groups
from seaborn._statistics import (
from seaborn.palettes import light_palette
from seaborn.axisgrid import FacetGrid, _facet_docs
def plot_bars(self, aggregator, dodge, gap, width, fill, color, capsize, err_kws, plot_kws):
    agg_var = {'x': 'y', 'y': 'x'}[self.orient]
    iter_vars = ['hue']
    ax = self.ax
    if self._hue_map.levels is None:
        dodge = False
    if dodge and capsize is not None:
        capsize = capsize / len(self._hue_map.levels)
    if not fill:
        plot_kws.setdefault('linewidth', 1.5 * mpl.rcParams['lines.linewidth'])
    err_kws.setdefault('linewidth', 1.5 * mpl.rcParams['lines.linewidth'])
    for sub_vars, sub_data in self.iter_data(iter_vars, from_comp_data=True, allow_empty=True):
        ax = self._get_axes(sub_vars)
        agg_data = sub_data if sub_data.empty else sub_data.groupby(self.orient).apply(aggregator, agg_var, **groupby_apply_include_groups(False)).reset_index()
        agg_data['width'] = width * self._native_width
        if dodge:
            self._dodge(sub_vars, agg_data)
        if gap:
            agg_data['width'] *= 1 - gap
        agg_data['edge'] = agg_data[self.orient] - agg_data['width'] / 2
        self._invert_scale(ax, agg_data)
        if self.orient == 'x':
            bar_func = ax.bar
            kws = dict(x=agg_data['edge'], height=agg_data['y'], width=agg_data['width'])
        else:
            bar_func = ax.barh
            kws = dict(y=agg_data['edge'], width=agg_data['x'], height=agg_data['width'])
        main_color = self._hue_map(sub_vars['hue']) if 'hue' in sub_vars else color
        kws['align'] = 'edge'
        if fill:
            kws.update(color=main_color, facecolor=main_color)
        else:
            kws.update(color=main_color, edgecolor=main_color, facecolor='none')
        bar_func(**{**kws, **plot_kws})
        if aggregator.error_method is not None:
            self.plot_errorbars(ax, agg_data, capsize, {'color': '.26' if fill else main_color, **err_kws})
    legend_artist = _get_patch_legend_artist(fill)
    self._configure_legend(ax, legend_artist, plot_kws)