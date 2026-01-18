from functools import partial
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cbook import normalize_kwargs
from ._base import (
from .utils import (
from ._compat import groupby_apply_include_groups
from ._statistics import EstimateAggregator, WeightedAggregator
from .axisgrid import FacetGrid, _facet_docs
from ._docstrings import DocstringComponents, _core_docs
def relplot(data=None, *, x=None, y=None, hue=None, size=None, style=None, units=None, weights=None, row=None, col=None, col_wrap=None, row_order=None, col_order=None, palette=None, hue_order=None, hue_norm=None, sizes=None, size_order=None, size_norm=None, markers=None, dashes=None, style_order=None, legend='auto', kind='scatter', height=5, aspect=1, facet_kws=None, **kwargs):
    if kind == 'scatter':
        Plotter = _ScatterPlotter
        func = scatterplot
        markers = True if markers is None else markers
    elif kind == 'line':
        Plotter = _LinePlotter
        func = lineplot
        dashes = True if dashes is None else dashes
    else:
        err = f'Plot kind {kind} not recognized'
        raise ValueError(err)
    if 'ax' in kwargs:
        msg = 'relplot is a figure-level function and does not accept the `ax` parameter. You may wish to try {}'.format(kind + 'plot')
        warnings.warn(msg, UserWarning)
        kwargs.pop('ax')
    variables = dict(x=x, y=y, hue=hue, size=size, style=style)
    if kind == 'line':
        variables['units'] = units
        variables['weight'] = weights
    else:
        if units is not None:
            msg = "The `units` parameter has no effect with kind='scatter'."
            warnings.warn(msg, stacklevel=2)
        if weights is not None:
            msg = "The `weights` parameter has no effect with kind='scatter'."
            warnings.warn(msg, stacklevel=2)
    p = Plotter(data=data, variables=variables, legend=legend)
    p.map_hue(palette=palette, order=hue_order, norm=hue_norm)
    p.map_size(sizes=sizes, order=size_order, norm=size_norm)
    p.map_style(markers=markers, dashes=dashes, order=style_order)
    if 'hue' in p.variables:
        palette = p._hue_map.lookup_table
        hue_order = p._hue_map.levels
        hue_norm = p._hue_map.norm
    else:
        palette = hue_order = hue_norm = None
    if 'size' in p.variables:
        sizes = p._size_map.lookup_table
        size_order = p._size_map.levels
        size_norm = p._size_map.norm
    if 'style' in p.variables:
        style_order = p._style_map.levels
        if markers:
            markers = {k: p._style_map(k, 'marker') for k in style_order}
        else:
            markers = None
        if dashes:
            dashes = {k: p._style_map(k, 'dashes') for k in style_order}
        else:
            dashes = None
    else:
        markers = dashes = style_order = None
    variables = p.variables
    plot_data = p.plot_data
    plot_kws = dict(palette=palette, hue_order=hue_order, hue_norm=hue_norm, sizes=sizes, size_order=size_order, size_norm=size_norm, markers=markers, dashes=dashes, style_order=style_order, legend=False)
    plot_kws.update(kwargs)
    if kind == 'scatter':
        plot_kws.pop('dashes')
    grid_variables = dict(x=x, y=y, row=row, col=col, hue=hue, size=size, style=style)
    if kind == 'line':
        grid_variables.update(units=units, weights=weights)
    p.assign_variables(data, grid_variables)
    plot_variables = {v: f'_{v}' for v in variables}
    if 'weight' in plot_variables:
        plot_variables['weights'] = plot_variables.pop('weight')
    plot_kws.update(plot_variables)
    for var in ['row', 'col']:
        if var in p.variables and p.variables[var] is None:
            p.variables[var] = f'_{var}_'
    grid_kws = {v: p.variables.get(v) for v in ['row', 'col']}
    new_cols = plot_variables.copy()
    new_cols.update(grid_kws)
    full_data = p.plot_data.rename(columns=new_cols)
    facet_kws = {} if facet_kws is None else facet_kws.copy()
    g = FacetGrid(data=full_data.dropna(axis=1, how='all'), **grid_kws, col_wrap=col_wrap, row_order=row_order, col_order=col_order, height=height, aspect=aspect, dropna=False, **facet_kws)
    g.map_dataframe(func, **plot_kws)
    g.set_axis_labels(variables.get('x') or '', variables.get('y') or '')
    if legend:
        p.plot_data = plot_data
        keys = ['c', 'color', 'alpha', 'm', 'marker']
        if kind == 'scatter':
            legend_artist = _scatter_legend_artist
            keys += ['s', 'facecolor', 'fc', 'edgecolor', 'ec', 'linewidth', 'lw']
        else:
            legend_artist = partial(mpl.lines.Line2D, xdata=[], ydata=[])
            keys += ['markersize', 'ms', 'markeredgewidth', 'mew', 'markeredgecolor', 'mec', 'linestyle', 'ls', 'linewidth', 'lw']
        common_kws = {k: v for k, v in kwargs.items() if k in keys}
        attrs = {'hue': 'color', 'style': None}
        if kind == 'scatter':
            attrs['size'] = 's'
        elif kind == 'line':
            attrs['size'] = 'linewidth'
        p.add_legend_data(g.axes.flat[0], legend_artist, common_kws, attrs)
        if p.legend_data:
            g.add_legend(legend_data=p.legend_data, label_order=p.legend_order, title=p.legend_title, adjust_subtitles=True)
    orig_cols = {f'_{k}': f'_{k}_' if v is None else v for k, v in variables.items()}
    grid_data = g.data.rename(columns=orig_cols)
    if data is not None and (x is not None or y is not None):
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        g.data = pd.merge(data, grid_data[grid_data.columns.difference(data.columns)], left_index=True, right_index=True)
    else:
        g.data = grid_data
    return g