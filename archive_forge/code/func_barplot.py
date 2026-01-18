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
def barplot(data=None, *, x=None, y=None, hue=None, order=None, hue_order=None, estimator='mean', errorbar=('ci', 95), n_boot=1000, seed=None, units=None, weights=None, orient=None, color=None, palette=None, saturation=0.75, fill=True, hue_norm=None, width=0.8, dodge='auto', gap=0, log_scale=None, native_scale=False, formatter=None, legend='auto', capsize=0, err_kws=None, ci=deprecated, errcolor=deprecated, errwidth=deprecated, ax=None, **kwargs):
    errorbar = utils._deprecate_ci(errorbar, ci)
    if estimator is len:
        estimator = 'size'
    p = _CategoricalAggPlotter(data=data, variables=dict(x=x, y=y, hue=hue, units=units, weight=weights), order=order, orient=orient, color=color, legend=legend)
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
    saturation = saturation if fill else 1
    p.map_hue(palette=palette, order=hue_order, norm=hue_norm, saturation=saturation)
    color = _default_color(ax.bar, hue, color, kwargs, saturation=saturation)
    agg_cls = WeightedAggregator if 'weight' in p.plot_data else EstimateAggregator
    aggregator = agg_cls(estimator, errorbar, n_boot=n_boot, seed=seed)
    err_kws = {} if err_kws is None else normalize_kwargs(err_kws, mpl.lines.Line2D)
    err_kws, capsize = p._err_kws_backcompat(err_kws, errcolor, errwidth, capsize)
    p.plot_bars(aggregator=aggregator, dodge=dodge, width=width, gap=gap, color=color, fill=fill, capsize=capsize, err_kws=err_kws, plot_kws=kwargs)
    p._add_axis_labels(ax)
    p._adjust_cat_axis(ax, axis=p.orient)
    return ax