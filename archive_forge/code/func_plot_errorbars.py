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
def plot_errorbars(self, ax, data, capsize, err_kws):
    var = {'x': 'y', 'y': 'x'}[self.orient]
    for row in data.to_dict('records'):
        row = dict(row)
        pos = np.array([row[self.orient], row[self.orient]])
        val = np.array([row[f'{var}min'], row[f'{var}max']])
        if capsize:
            cw = capsize * self._native_width / 2
            scl, inv = _get_transform_functions(ax, self.orient)
            cap = (inv(scl(pos[0]) - cw), inv(scl(pos[1]) + cw))
            pos = np.concatenate([[*cap, np.nan], pos, [np.nan, *cap]])
            val = np.concatenate([[val[0], val[0], np.nan], val, [np.nan, val[-1], val[-1]]])
        if self.orient == 'x':
            args = (pos, val)
        else:
            args = (val, pos)
        ax.plot(*args, **err_kws)