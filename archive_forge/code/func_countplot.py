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
def countplot(data=None, *, x=None, y=None, hue=None, order=None, hue_order=None, orient=None, color=None, palette=None, saturation=0.75, fill=True, hue_norm=None, stat='count', width=0.8, dodge='auto', gap=0, log_scale=None, native_scale=False, formatter=None, legend='auto', ax=None, **kwargs):
    if x is None and y is not None:
        orient = 'y'
        x = 1 if list(y) else None
    elif x is not None and y is None:
        orient = 'x'
        y = 1 if list(x) else None
    elif x is not None and y is not None:
        raise TypeError('Cannot pass values for both `x` and `y`.')
    p = _CategoricalAggPlotter(data=data, variables=dict(x=x, y=y, hue=hue), order=order, orient=orient, color=color, legend=legend)
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
    color = _default_color(ax.bar, hue, color, kwargs, saturation)
    count_axis = {'x': 'y', 'y': 'x'}[p.orient]
    if p.input_format == 'wide':
        p.plot_data[count_axis] = 1
    _check_argument('stat', ['count', 'percent', 'probability', 'proportion'], stat)
    p.variables[count_axis] = stat
    if stat != 'count':
        denom = 100 if stat == 'percent' else 1
        p.plot_data[count_axis] /= len(p.plot_data) / denom
    aggregator = EstimateAggregator('sum', errorbar=None)
    p.plot_bars(aggregator=aggregator, dodge=dodge, width=width, gap=gap, color=color, fill=fill, capsize=0, err_kws={}, plot_kws=kwargs)
    p._add_axis_labels(ax)
    p._adjust_cat_axis(ax, axis=p.orient)
    return ax