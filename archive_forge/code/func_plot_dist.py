import matplotlib.pyplot as plt
import numpy as np
from ....stats.density_utils import get_bins, histogram
from ...kdeplot import plot_kde
from ...plot_utils import (
from .. import show_layout
from . import backend_kwarg_defaults, create_axes_grid
def plot_dist(values, values2, color, kind, cumulative, label, rotated, rug, bw, quantiles, contour, fill_last, figsize, textsize, plot_kwargs, fill_kwargs, rug_kwargs, contour_kwargs, contourf_kwargs, pcolormesh_kwargs, hist_kwargs, is_circular, ax, backend_kwargs, show):
    """Bokeh distplot."""
    backend_kwargs = _init_kwargs_dict(backend_kwargs)
    backend_kwargs = {**backend_kwarg_defaults(), **backend_kwargs}
    figsize, *_ = _scale_fig_size(figsize, textsize)
    color = vectorized_to_hex(color)
    hist_kwargs = _init_kwargs_dict(hist_kwargs)
    if kind == 'hist':
        hist_kwargs.setdefault('cumulative', cumulative)
        hist_kwargs.setdefault('fill_color', color)
        hist_kwargs.setdefault('line_color', color)
        hist_kwargs.setdefault('line_alpha', 0)
        if label is not None:
            hist_kwargs.setdefault('legend_label', str(label))
    if ax is None:
        ax = create_axes_grid(1, figsize=figsize, squeeze=True, polar=is_circular, backend_kwargs=backend_kwargs)
    if kind == 'auto':
        kind = 'hist' if values.dtype.kind == 'i' else 'kde'
    if kind == 'hist':
        _histplot_bokeh_op(values=values, values2=values2, rotated=rotated, ax=ax, hist_kwargs=hist_kwargs, is_circular=is_circular)
    elif kind == 'kde':
        plot_kwargs = _init_kwargs_dict(plot_kwargs)
        if color is None:
            color = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
        plot_kwargs.setdefault('line_color', color)
        legend = label is not None
        plot_kde(values, values2, cumulative=cumulative, rug=rug, label=label, bw=bw, is_circular=is_circular, quantiles=quantiles, rotated=rotated, contour=contour, legend=legend, fill_last=fill_last, plot_kwargs=plot_kwargs, fill_kwargs=fill_kwargs, rug_kwargs=rug_kwargs, contour_kwargs=contour_kwargs, contourf_kwargs=contourf_kwargs, pcolormesh_kwargs=pcolormesh_kwargs, ax=ax, backend='bokeh', backend_kwargs={}, show=False)
    else:
        raise TypeError(f'Invalid "kind":{kind}. Select from {{"auto","kde","hist"}}')
    show_layout(ax, show)
    return ax