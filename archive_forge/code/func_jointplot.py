from __future__ import annotations
from itertools import product
from inspect import signature
import warnings
from textwrap import dedent
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from ._base import VectorPlotter, variable_type, categorical_order
from ._core.data import handle_data_source
from ._compat import share_axis, get_legend_handles
from . import utils
from .utils import (
from .palettes import color_palette, blend_palette
from ._docstrings import (
def jointplot(data=None, *, x=None, y=None, hue=None, kind='scatter', height=6, ratio=5, space=0.2, dropna=False, xlim=None, ylim=None, color=None, palette=None, hue_order=None, hue_norm=None, marginal_ticks=False, joint_kws=None, marginal_kws=None, **kwargs):
    from .relational import scatterplot
    from .regression import regplot, residplot
    from .distributions import histplot, kdeplot, _freedman_diaconis_bins
    if kwargs.pop('ax', None) is not None:
        msg = 'Ignoring `ax`; jointplot is a figure-level function.'
        warnings.warn(msg, UserWarning, stacklevel=2)
    joint_kws = {} if joint_kws is None else joint_kws.copy()
    joint_kws.update(kwargs)
    marginal_kws = {} if marginal_kws is None else marginal_kws.copy()
    distplot_keys = ['rug', 'fit', 'hist_kws', 'norm_histhist_kws', 'rug_kws']
    unused_keys = []
    for key in distplot_keys:
        if key in marginal_kws:
            unused_keys.append(key)
            marginal_kws.pop(key)
    if unused_keys and kind != 'kde':
        msg = 'The marginal plotting function has changed to `histplot`, which does not accept the following argument(s): {}.'.format(', '.join(unused_keys))
        warnings.warn(msg, UserWarning)
    plot_kinds = ['scatter', 'hist', 'hex', 'kde', 'reg', 'resid']
    _check_argument('kind', plot_kinds, kind)
    if hue is not None and kind in ['hex', 'reg', 'resid']:
        msg = f"Use of `hue` with `kind='{kind}'` is not currently supported."
        raise ValueError(msg)
    if color is None:
        color = 'C0'
    color_rgb = mpl.colors.colorConverter.to_rgb(color)
    colors = [set_hls_values(color_rgb, l=val) for val in np.linspace(1, 0, 12)]
    cmap = blend_palette(colors, as_cmap=True)
    if kind == 'hex':
        dropna = True
    grid = JointGrid(data=data, x=x, y=y, hue=hue, palette=palette, hue_order=hue_order, hue_norm=hue_norm, dropna=dropna, height=height, ratio=ratio, space=space, xlim=xlim, ylim=ylim, marginal_ticks=marginal_ticks)
    if grid.hue is not None:
        marginal_kws.setdefault('legend', False)
    if kind.startswith('scatter'):
        joint_kws.setdefault('color', color)
        grid.plot_joint(scatterplot, **joint_kws)
        if grid.hue is None:
            marg_func = histplot
        else:
            marg_func = kdeplot
            marginal_kws.setdefault('warn_singular', False)
            marginal_kws.setdefault('fill', True)
        marginal_kws.setdefault('color', color)
        grid.plot_marginals(marg_func, **marginal_kws)
    elif kind.startswith('hist'):
        joint_kws.setdefault('color', color)
        grid.plot_joint(histplot, **joint_kws)
        marginal_kws.setdefault('kde', False)
        marginal_kws.setdefault('color', color)
        marg_x_kws = marginal_kws.copy()
        marg_y_kws = marginal_kws.copy()
        pair_keys = ('bins', 'binwidth', 'binrange')
        for key in pair_keys:
            if isinstance(joint_kws.get(key), tuple):
                x_val, y_val = joint_kws[key]
                marg_x_kws.setdefault(key, x_val)
                marg_y_kws.setdefault(key, y_val)
        histplot(data=data, x=x, hue=hue, **marg_x_kws, ax=grid.ax_marg_x)
        histplot(data=data, y=y, hue=hue, **marg_y_kws, ax=grid.ax_marg_y)
    elif kind.startswith('kde'):
        joint_kws.setdefault('color', color)
        joint_kws.setdefault('warn_singular', False)
        grid.plot_joint(kdeplot, **joint_kws)
        marginal_kws.setdefault('color', color)
        if 'fill' in joint_kws:
            marginal_kws.setdefault('fill', joint_kws['fill'])
        grid.plot_marginals(kdeplot, **marginal_kws)
    elif kind.startswith('hex'):
        x_bins = min(_freedman_diaconis_bins(grid.x), 50)
        y_bins = min(_freedman_diaconis_bins(grid.y), 50)
        gridsize = int(np.mean([x_bins, y_bins]))
        joint_kws.setdefault('gridsize', gridsize)
        joint_kws.setdefault('cmap', cmap)
        grid.plot_joint(plt.hexbin, **joint_kws)
        marginal_kws.setdefault('kde', False)
        marginal_kws.setdefault('color', color)
        grid.plot_marginals(histplot, **marginal_kws)
    elif kind.startswith('reg'):
        marginal_kws.setdefault('color', color)
        marginal_kws.setdefault('kde', True)
        grid.plot_marginals(histplot, **marginal_kws)
        joint_kws.setdefault('color', color)
        grid.plot_joint(regplot, **joint_kws)
    elif kind.startswith('resid'):
        joint_kws.setdefault('color', color)
        grid.plot_joint(residplot, **joint_kws)
        x, y = grid.ax_joint.collections[0].get_offsets().T
        marginal_kws.setdefault('color', color)
        histplot(x=x, hue=hue, ax=grid.ax_marg_x, **marginal_kws)
        histplot(y=y, hue=hue, ax=grid.ax_marg_y, **marginal_kws)
    plt.sca(grid.ax_joint)
    return grid