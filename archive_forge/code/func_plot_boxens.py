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
def plot_boxens(self, width, dodge, gap, fill, color, linecolor, linewidth, width_method, k_depth, outlier_prop, trust_alpha, showfliers, box_kws, flier_kws, line_kws, plot_kws):
    iter_vars = [self.orient, 'hue']
    value_var = {'x': 'y', 'y': 'x'}[self.orient]
    estimator = LetterValues(k_depth, outlier_prop, trust_alpha)
    width_method_options = ['exponential', 'linear', 'area']
    _check_argument('width_method', width_method_options, width_method)
    box_kws = plot_kws if box_kws is None else {**plot_kws, **box_kws}
    flier_kws = {} if flier_kws is None else flier_kws.copy()
    line_kws = {} if line_kws is None else line_kws.copy()
    if linewidth is None:
        if fill:
            linewidth = 0.5 * mpl.rcParams['lines.linewidth']
        else:
            linewidth = mpl.rcParams['lines.linewidth']
    ax = self.ax
    for sub_vars, sub_data in self.iter_data(iter_vars, from_comp_data=True, allow_empty=False):
        ax = self._get_axes(sub_vars)
        _, inv_ori = _get_transform_functions(ax, self.orient)
        _, inv_val = _get_transform_functions(ax, value_var)
        lv_data = estimator(sub_data[value_var])
        n = lv_data['k'] * 2 - 1
        vals = lv_data['values']
        pos_data = pd.DataFrame({self.orient: [sub_vars[self.orient]], 'width': [width * self._native_width]})
        if dodge:
            self._dodge(sub_vars, pos_data)
        if gap:
            pos_data['width'] *= 1 - gap
        levels = lv_data['levels']
        exponent = (levels - 1 - lv_data['k']).astype(float)
        if width_method == 'linear':
            rel_widths = levels + 1
        elif width_method == 'exponential':
            rel_widths = 2 ** exponent
        elif width_method == 'area':
            tails = levels < lv_data['k'] - 1
            rel_widths = 2 ** (exponent - tails) / np.diff(lv_data['values'])
        center = pos_data[self.orient].item()
        widths = rel_widths / rel_widths.max() * pos_data['width'].item()
        box_vals = inv_val(vals)
        box_pos = inv_ori(center - widths / 2)
        box_heights = inv_val(vals[1:]) - inv_val(vals[:-1])
        box_widths = inv_ori(center + widths / 2) - inv_ori(center - widths / 2)
        maincolor = self._hue_map(sub_vars['hue']) if 'hue' in sub_vars else color
        flier_colors = {'facecolor': 'none', 'edgecolor': '.45' if fill else maincolor}
        if fill:
            cmap = light_palette(maincolor, as_cmap=True)
            boxcolors = cmap(2 ** ((exponent + 2) / 3))
        else:
            boxcolors = maincolor
        boxen = []
        for i in range(n):
            if self.orient == 'x':
                xy = (box_pos[i], box_vals[i])
                w, h = (box_widths[i], box_heights[i])
            else:
                xy = (box_vals[i], box_pos[i])
                w, h = (box_heights[i], box_widths[i])
            boxen.append(Rectangle(xy, w, h))
        if fill:
            box_colors = {'facecolors': boxcolors, 'edgecolors': linecolor}
        else:
            box_colors = {'facecolors': 'none', 'edgecolors': boxcolors}
        collection_kws = {**box_colors, 'linewidth': linewidth, **box_kws}
        ax.add_collection(PatchCollection(boxen, **collection_kws), autolim=False)
        ax.update_datalim(np.column_stack([box_vals, box_vals]), updatex=self.orient == 'y', updatey=self.orient == 'x')
        med = lv_data['median']
        hw = pos_data['width'].item() / 2
        if self.orient == 'x':
            x, y = (inv_ori([center - hw, center + hw]), inv_val([med, med]))
        else:
            x, y = (inv_val([med, med]), inv_ori([center - hw, center + hw]))
        default_kws = {'color': linecolor if fill else maincolor, 'solid_capstyle': 'butt', 'linewidth': 1.25 * linewidth}
        ax.plot(x, y, **{**default_kws, **line_kws})
        if showfliers:
            vals = inv_val(lv_data['fliers'])
            pos = np.full(len(vals), inv_ori(pos_data[self.orient].item()))
            x, y = (pos, vals) if self.orient == 'x' else (vals, pos)
            ax.scatter(x, y, **{**flier_colors, 's': 25, **flier_kws})
    ax.autoscale_view(scalex=self.orient == 'y', scaley=self.orient == 'x')
    legend_artist = _get_patch_legend_artist(fill)
    common_kws = {**box_kws, 'linewidth': linewidth, 'edgecolor': linecolor}
    self._configure_legend(ax, legend_artist, common_kws)