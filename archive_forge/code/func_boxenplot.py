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
def boxenplot(data=None, *, x=None, y=None, hue=None, order=None, hue_order=None, orient=None, color=None, palette=None, saturation=0.75, fill=True, dodge='auto', width=0.8, gap=0, linewidth=None, linecolor=None, width_method='exponential', k_depth='tukey', outlier_prop=0.007, trust_alpha=0.05, showfliers=True, hue_norm=None, log_scale=None, native_scale=False, formatter=None, legend='auto', scale=deprecated, box_kws=None, flier_kws=None, line_kws=None, ax=None, **kwargs):
    p = _CategoricalPlotter(data=data, variables=dict(x=x, y=y, hue=hue), order=order, orient=orient, color=color, legend=legend)
    if ax is None:
        ax = plt.gca()
    if p.plot_data.empty:
        return ax
    if dodge == 'auto':
        dodge = p._dodge_needed()
    if p.var_types.get(p.orient) == 'categorical' or not native_scale:
        p.scale_categorical(p.orient, order=order, formatter=formatter)
    p._attach(ax, log_scale=log_scale)
    hue_order = p._palette_without_hue_backcompat(palette, hue_order)
    palette, hue_order = p._hue_backcompat(color, palette, hue_order)
    width_method = p._boxen_scale_backcompat(scale, width_method)
    saturation = saturation if fill else 1
    p.map_hue(palette=palette, order=hue_order, norm=hue_norm, saturation=saturation)
    color = _default_color(ax.fill_between, hue, color, {}, saturation=saturation)
    linecolor = p._complement_color(linecolor, color, p._hue_map)
    p.plot_boxens(width=width, dodge=dodge, gap=gap, fill=fill, color=color, linecolor=linecolor, linewidth=linewidth, width_method=width_method, k_depth=k_depth, outlier_prop=outlier_prop, trust_alpha=trust_alpha, showfliers=showfliers, box_kws=box_kws, flier_kws=flier_kws, line_kws=line_kws, plot_kws=kwargs)
    p._add_axis_labels(ax)
    p._adjust_cat_axis(ax, axis=p.orient)
    return ax