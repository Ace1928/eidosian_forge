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
def stripplot(data=None, *, x=None, y=None, hue=None, order=None, hue_order=None, jitter=True, dodge=False, orient=None, color=None, palette=None, size=5, edgecolor=default, linewidth=0, hue_norm=None, log_scale=None, native_scale=False, formatter=None, legend='auto', ax=None, **kwargs):
    p = _CategoricalPlotter(data=data, variables=dict(x=x, y=y, hue=hue), order=order, orient=orient, color=color, legend=legend)
    if ax is None:
        ax = plt.gca()
    if p.plot_data.empty:
        return ax
    if p.var_types.get(p.orient) == 'categorical' or not native_scale:
        p.scale_categorical(p.orient, order=order, formatter=formatter)
    p._attach(ax, log_scale=log_scale)
    hue_order = p._palette_without_hue_backcompat(palette, hue_order)
    palette, hue_order = p._hue_backcompat(color, palette, hue_order)
    p.map_hue(palette=palette, order=hue_order, norm=hue_norm)
    color = _default_color(ax.scatter, hue, color, kwargs)
    edgecolor = p._complement_color(edgecolor, color, p._hue_map)
    kwargs.setdefault('zorder', 3)
    size = kwargs.get('s', size)
    kwargs.update(s=size ** 2, edgecolor=edgecolor, linewidth=linewidth)
    p.plot_strips(jitter=jitter, dodge=dodge, color=color, plot_kws=kwargs)
    p._add_axis_labels(ax)
    p._adjust_cat_axis(ax, axis=p.orient)
    return ax