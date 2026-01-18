import warnings
import bokeh.plotting as bkp
import numpy as np
from bokeh.models import ColumnDataSource
from bokeh.models.annotations import Title
from bokeh.models.glyphs import Scatter
from ....rcparams import _validate_bokeh_marker, rcParams
from ...plot_utils import _scale_fig_size, color_from_dim, vectorized_to_hex
from .. import show_layout
from . import backend_kwarg_defaults, create_axes_grid
def plot_elpd(ax, models, pointwise_data, numvars, figsize, textsize, plot_kwargs, xlabels, coord_labels, xdata, threshold, legend, color, backend_kwargs, show):
    """Bokeh elpd plot."""
    if backend_kwargs is None:
        backend_kwargs = {}
    backend_kwargs = {**backend_kwarg_defaults(('dpi', 'plot.bokeh.figure.dpi')), **backend_kwargs}
    plot_kwargs = {} if plot_kwargs is None else plot_kwargs
    plot_kwargs.setdefault('marker', rcParams['plot.bokeh.marker'])
    if isinstance(color, str) and color in pointwise_data[0].dims:
        colors, _ = color_from_dim(pointwise_data[0], color)
        plot_kwargs.setdefault('color', vectorized_to_hex(colors))
    plot_kwargs.setdefault('color', vectorized_to_hex(color))
    pointwise_data = [pointwise.values.flatten() for pointwise in pointwise_data]
    if numvars == 2:
        figsize, _, _, _, _, markersize = _scale_fig_size(figsize, textsize, numvars - 1, numvars - 1)
        plot_kwargs.setdefault('s', markersize)
        if ax is None:
            ax = create_axes_grid(1, figsize=figsize, squeeze=True, backend_kwargs=backend_kwargs)
        ydata = pointwise_data[0] - pointwise_data[1]
        _plot_atomic_elpd(ax, xdata, ydata, *models, threshold, coord_labels, xlabels, True, True, plot_kwargs)
    else:
        max_plots = numvars ** 2 if rcParams['plot.max_subplots'] is None else rcParams['plot.max_subplots']
        vars_to_plot = np.sum(np.arange(numvars).cumsum() < max_plots)
        if vars_to_plot < numvars:
            warnings.warn("rcParams['plot.max_subplots'] ({max_plots}) is smaller than the number of resulting ELPD pairwise plots with these variables, generating only a {side}x{side} grid".format(max_plots=max_plots, side=vars_to_plot), UserWarning)
            numvars = vars_to_plot
        figsize, _, _, _, _, markersize = _scale_fig_size(figsize, textsize, numvars - 2, numvars - 2)
        plot_kwargs.setdefault('s', markersize)
        if ax is None:
            dpi = backend_kwargs.pop('dpi')
            ax = []
            for row in range(numvars - 1):
                ax_row = []
                for col in range(numvars - 1):
                    if row == 0 and col == 0:
                        ax_first = bkp.figure(width=int(figsize[0] / (numvars - 1) * dpi), height=int(figsize[1] / (numvars - 1) * dpi), **backend_kwargs)
                        ax_row.append(ax_first)
                    elif row < col:
                        ax_row.append(None)
                    else:
                        ax_row.append(bkp.figure(width=int(figsize[0] / (numvars - 1) * dpi), height=int(figsize[1] / (numvars - 1) * dpi), x_range=ax_first.x_range, y_range=ax_first.y_range, **backend_kwargs))
                ax.append(ax_row)
            ax = np.array(ax)
        for i in range(0, numvars - 1):
            var1 = pointwise_data[i]
            for j in range(0, numvars - 1):
                if j < i:
                    continue
                var2 = pointwise_data[j + 1]
                ydata = var1 - var2
                _plot_atomic_elpd(ax[j, i], xdata, ydata, models[i], models[j + 1], threshold, coord_labels, xlabels, j == numvars - 2, i == 0, plot_kwargs)
    show_layout(ax, show)
    return ax