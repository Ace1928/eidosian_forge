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
def plot_swarms(self, dodge, color, warn_thresh, plot_kws):
    width = 0.8 * self._native_width
    offsets = self._nested_offsets(width, dodge)
    iter_vars = [self.orient]
    if dodge:
        iter_vars.append('hue')
    ax = self.ax
    point_collections = {}
    dodge_move = 0
    if 'marker' in plot_kws and (not MarkerStyle(plot_kws['marker']).is_filled()):
        plot_kws.pop('edgecolor', None)
    for sub_vars, sub_data in self.iter_data(iter_vars, from_comp_data=True, allow_empty=True):
        ax = self._get_axes(sub_vars)
        if offsets is not None:
            dodge_move = offsets[sub_data['hue'].map(self._hue_map.levels.index)]
        if not sub_data.empty:
            sub_data[self.orient] = sub_data[self.orient] + dodge_move
        self._invert_scale(ax, sub_data)
        points = ax.scatter(sub_data['x'], sub_data['y'], color=color, **plot_kws)
        if 'hue' in self.variables:
            points.set_facecolors(self._hue_map(sub_data['hue']))
        if not sub_data.empty:
            point_collections[ax, sub_data[self.orient].iloc[0]] = points
    beeswarm = Beeswarm(width=width, orient=self.orient, warn_thresh=warn_thresh)
    for (ax, center), points in point_collections.items():
        if points.get_offsets().shape[0] > 1:

            def draw(points, renderer, *, center=center):
                beeswarm(points, center)
                if self.orient == 'y':
                    scalex = False
                    scaley = ax.get_autoscaley_on()
                else:
                    scalex = ax.get_autoscalex_on()
                    scaley = False
                fixed_scale = self.var_types[self.orient] == 'categorical'
                ax.update_datalim(points.get_datalim(ax.transData))
                if not fixed_scale and (scalex or scaley):
                    ax.autoscale_view(scalex=scalex, scaley=scaley)
                super(points.__class__, points).draw(renderer)
            points.draw = draw.__get__(points)
    _draw_figure(ax.figure)
    self._configure_legend(ax, _scatter_legend_artist, plot_kws)