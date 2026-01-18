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
def catplot(data=None, *, x=None, y=None, hue=None, row=None, col=None, kind='strip', estimator='mean', errorbar=('ci', 95), n_boot=1000, seed=None, units=None, weights=None, order=None, hue_order=None, row_order=None, col_order=None, col_wrap=None, height=5, aspect=1, log_scale=None, native_scale=False, formatter=None, orient=None, color=None, palette=None, hue_norm=None, legend='auto', legend_out=True, sharex=True, sharey=True, margin_titles=False, facet_kws=None, ci=deprecated, **kwargs):
    if 'ax' in kwargs:
        msg = f'catplot is a figure-level function and does not accept target axes. You may wish to try {kind}plot'
        warnings.warn(msg, UserWarning)
        kwargs.pop('ax')
    desaturated_kinds = ['bar', 'count', 'box', 'violin', 'boxen']
    undodged_kinds = ['strip', 'swarm', 'point']
    if kind in ['bar', 'point', 'count']:
        Plotter = _CategoricalAggPlotter
    else:
        Plotter = _CategoricalPlotter
    if kind == 'count':
        if x is None and y is not None:
            orient = 'y'
            x = 1
        elif x is not None and y is None:
            orient = 'x'
            y = 1
        elif x is not None and y is not None:
            raise ValueError('Cannot pass values for both `x` and `y`.')
    p = Plotter(data=data, variables=dict(x=x, y=y, hue=hue, row=row, col=col, units=units, weight=weights), order=order, orient=orient, color='C0' if kind == 'point' and palette is None and (color is None) else color, legend=legend)
    for var in ['row', 'col']:
        if var in p.variables and p.variables[var] is None:
            p.variables[var] = f'_{var}_'
    facet_data = p.plot_data.rename(columns=p.variables)
    facet_data = facet_data.loc[:, ~facet_data.columns.duplicated()]
    col_name = p.variables.get('col', None)
    row_name = p.variables.get('row', None)
    if facet_kws is None:
        facet_kws = {}
    g = FacetGrid(data=facet_data, row=row_name, col=col_name, col_wrap=col_wrap, row_order=row_order, col_order=col_order, sharex=sharex, sharey=sharey, legend_out=legend_out, margin_titles=margin_titles, height=height, aspect=aspect, **facet_kws)
    has_xy_data = p.has_xy_data
    if not native_scale or p.var_types[p.orient] == 'categorical':
        p.scale_categorical(p.orient, order=order, formatter=formatter)
    p._attach(g, log_scale=log_scale)
    if not has_xy_data:
        return g
    hue_order = p._palette_without_hue_backcompat(palette, hue_order)
    palette, hue_order = p._hue_backcompat(color, palette, hue_order)
    errorbar = utils._deprecate_ci(errorbar, ci)
    saturation = kwargs.pop('saturation', 0.75 if kind in desaturated_kinds and kwargs.get('fill', True) else 1)
    p.map_hue(palette=palette, order=hue_order, norm=hue_norm, saturation=saturation)
    if hue is None:
        color = 'C0' if color is None else color
        if saturation < 1:
            color = desaturate(color, saturation)
    if kind in ['strip', 'swarm']:
        kwargs = normalize_kwargs(kwargs, mpl.collections.PathCollection)
        kwargs['edgecolor'] = p._complement_color(kwargs.pop('edgecolor', default), color, p._hue_map)
    width = kwargs.pop('width', 0.8)
    dodge = kwargs.pop('dodge', False if kind in undodged_kinds else 'auto')
    if dodge == 'auto':
        dodge = p._dodge_needed()
    if 'weight' in p.plot_data:
        if kind not in ['bar', 'point']:
            msg = f'The `weights` parameter has no effect with kind={kind!r}.'
            warnings.warn(msg, stacklevel=2)
        agg_cls = WeightedAggregator
    else:
        agg_cls = EstimateAggregator
    if kind == 'strip':
        jitter = kwargs.pop('jitter', True)
        plot_kws = kwargs.copy()
        plot_kws.setdefault('zorder', 3)
        plot_kws.setdefault('linewidth', 0)
        if 's' not in plot_kws:
            plot_kws['s'] = plot_kws.pop('size', 5) ** 2
        p.plot_strips(jitter=jitter, dodge=dodge, color=color, plot_kws=plot_kws)
    elif kind == 'swarm':
        warn_thresh = kwargs.pop('warn_thresh', 0.05)
        plot_kws = kwargs.copy()
        plot_kws.setdefault('zorder', 3)
        if 's' not in plot_kws:
            plot_kws['s'] = plot_kws.pop('size', 5) ** 2
        if plot_kws.setdefault('linewidth', 0) is None:
            plot_kws['linewidth'] = np.sqrt(plot_kws['s']) / 10
        p.plot_swarms(dodge=dodge, color=color, warn_thresh=warn_thresh, plot_kws=plot_kws)
    elif kind == 'box':
        plot_kws = kwargs.copy()
        gap = plot_kws.pop('gap', 0)
        fill = plot_kws.pop('fill', True)
        whis = plot_kws.pop('whis', 1.5)
        linewidth = plot_kws.pop('linewidth', None)
        fliersize = plot_kws.pop('fliersize', 5)
        linecolor = p._complement_color(plot_kws.pop('linecolor', 'auto'), color, p._hue_map)
        p.plot_boxes(width=width, dodge=dodge, gap=gap, fill=fill, whis=whis, color=color, linecolor=linecolor, linewidth=linewidth, fliersize=fliersize, plot_kws=plot_kws)
    elif kind == 'violin':
        plot_kws = kwargs.copy()
        gap = plot_kws.pop('gap', 0)
        fill = plot_kws.pop('fill', True)
        split = plot_kws.pop('split', False)
        inner = plot_kws.pop('inner', 'box')
        density_norm = plot_kws.pop('density_norm', 'area')
        common_norm = plot_kws.pop('common_norm', False)
        scale = plot_kws.pop('scale', deprecated)
        scale_hue = plot_kws.pop('scale_hue', deprecated)
        density_norm, common_norm = p._violin_scale_backcompat(scale, scale_hue, density_norm, common_norm)
        bw_method = p._violin_bw_backcompat(plot_kws.pop('bw', deprecated), plot_kws.pop('bw_method', 'scott'))
        kde_kws = dict(cut=plot_kws.pop('cut', 2), gridsize=plot_kws.pop('gridsize', 100), bw_adjust=plot_kws.pop('bw_adjust', 1), bw_method=bw_method)
        inner_kws = plot_kws.pop('inner_kws', {}).copy()
        linewidth = plot_kws.pop('linewidth', None)
        linecolor = plot_kws.pop('linecolor', 'auto')
        linecolor = p._complement_color(linecolor, color, p._hue_map)
        p.plot_violins(width=width, dodge=dodge, gap=gap, split=split, color=color, fill=fill, linecolor=linecolor, linewidth=linewidth, inner=inner, density_norm=density_norm, common_norm=common_norm, kde_kws=kde_kws, inner_kws=inner_kws, plot_kws=plot_kws)
    elif kind == 'boxen':
        plot_kws = kwargs.copy()
        gap = plot_kws.pop('gap', 0)
        fill = plot_kws.pop('fill', True)
        linecolor = plot_kws.pop('linecolor', 'auto')
        linewidth = plot_kws.pop('linewidth', None)
        k_depth = plot_kws.pop('k_depth', 'tukey')
        width_method = plot_kws.pop('width_method', 'exponential')
        outlier_prop = plot_kws.pop('outlier_prop', 0.007)
        trust_alpha = plot_kws.pop('trust_alpha', 0.05)
        showfliers = plot_kws.pop('showfliers', True)
        box_kws = plot_kws.pop('box_kws', {})
        flier_kws = plot_kws.pop('flier_kws', {})
        line_kws = plot_kws.pop('line_kws', {})
        if 'scale' in plot_kws:
            width_method = p._boxen_scale_backcompat(plot_kws['scale'], width_method)
        linecolor = p._complement_color(linecolor, color, p._hue_map)
        p.plot_boxens(width=width, dodge=dodge, gap=gap, fill=fill, color=color, linecolor=linecolor, linewidth=linewidth, width_method=width_method, k_depth=k_depth, outlier_prop=outlier_prop, trust_alpha=trust_alpha, showfliers=showfliers, box_kws=box_kws, flier_kws=flier_kws, line_kws=line_kws, plot_kws=plot_kws)
    elif kind == 'point':
        aggregator = agg_cls(estimator, errorbar, n_boot=n_boot, seed=seed)
        markers = kwargs.pop('markers', default)
        linestyles = kwargs.pop('linestyles', default)
        p._point_kwargs_backcompat(kwargs.pop('scale', deprecated), kwargs.pop('join', deprecated), kwargs)
        err_kws, capsize = p._err_kws_backcompat(normalize_kwargs(kwargs.pop('err_kws', {}), mpl.lines.Line2D), None, errwidth=kwargs.pop('errwidth', deprecated), capsize=kwargs.pop('capsize', 0))
        p.plot_points(aggregator=aggregator, markers=markers, linestyles=linestyles, dodge=dodge, color=color, capsize=capsize, err_kws=err_kws, plot_kws=kwargs)
    elif kind == 'bar':
        aggregator = agg_cls(estimator, errorbar, n_boot=n_boot, seed=seed)
        err_kws, capsize = p._err_kws_backcompat(normalize_kwargs(kwargs.pop('err_kws', {}), mpl.lines.Line2D), errcolor=kwargs.pop('errcolor', deprecated), errwidth=kwargs.pop('errwidth', deprecated), capsize=kwargs.pop('capsize', 0))
        gap = kwargs.pop('gap', 0)
        fill = kwargs.pop('fill', True)
        p.plot_bars(aggregator=aggregator, dodge=dodge, width=width, gap=gap, color=color, fill=fill, capsize=capsize, err_kws=err_kws, plot_kws=kwargs)
    elif kind == 'count':
        aggregator = EstimateAggregator('sum', errorbar=None)
        count_axis = {'x': 'y', 'y': 'x'}[p.orient]
        p.plot_data[count_axis] = 1
        stat_options = ['count', 'percent', 'probability', 'proportion']
        stat = _check_argument('stat', stat_options, kwargs.pop('stat', 'count'))
        p.variables[count_axis] = stat
        if stat != 'count':
            denom = 100 if stat == 'percent' else 1
            p.plot_data[count_axis] /= len(p.plot_data) / denom
        gap = kwargs.pop('gap', 0)
        fill = kwargs.pop('fill', True)
        p.plot_bars(aggregator=aggregator, dodge=dodge, width=width, gap=gap, color=color, fill=fill, capsize=0, err_kws={}, plot_kws=kwargs)
    else:
        msg = f"Invalid `kind`: {kind!r}. Options are 'strip', 'swarm', 'box', 'boxen', 'violin', 'bar', 'count', and 'point'."
        raise ValueError(msg)
    for ax in g.axes.flat:
        p._adjust_cat_axis(ax, axis=p.orient)
    g.set_axis_labels(p.variables.get('x'), p.variables.get('y'))
    g.set_titles()
    g.tight_layout()
    for ax in g.axes.flat:
        g._update_legend_data(ax)
        ax.legend_ = None
    if legend == 'auto':
        show_legend = not p._redundant_hue and p.input_format != 'wide'
    else:
        show_legend = bool(legend)
    if show_legend:
        g.add_legend(title=p.variables.get('hue'), label_order=hue_order)
    if data is not None:
        g.data = data
    return g