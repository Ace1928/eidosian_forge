from collections import defaultdict
from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np
from bokeh.models.annotations import Legend, Title
from ....stats import hdi
from ....stats.density_utils import get_bins, histogram, kde
from ...plot_utils import _scale_fig_size, calculate_point_estimate, vectorized_to_hex
from .. import show_layout
from . import backend_kwarg_defaults, create_axes_grid
def plot_density(ax, all_labels, to_plot, colors, bw, circular, figsize, length_plotters, rows, cols, textsize, labeller, hdi_prob, point_estimate, hdi_markers, outline, shade, n_data, data_labels, backend_kwargs, show):
    """Bokeh density plot."""
    if backend_kwargs is None:
        backend_kwargs = {}
    backend_kwargs = {**backend_kwarg_defaults(), **backend_kwargs}
    if colors == 'cycle':
        colors = [prop for _, prop in zip(range(n_data), cycle(plt.rcParams['axes.prop_cycle'].by_key()['color']))]
    elif isinstance(colors, str):
        colors = [colors for _ in range(n_data)]
    colors = vectorized_to_hex(colors)
    figsize, _, _, _, line_width, markersize = _scale_fig_size(figsize, textsize, rows, cols)
    if ax is None:
        ax = create_axes_grid(length_plotters, rows, cols, figsize=figsize, squeeze=False, backend_kwargs=backend_kwargs)
    else:
        ax = np.atleast_2d(ax)
    axis_map = dict(zip(all_labels, (item for item in ax.flatten() if item is not None)))
    if data_labels is None:
        data_labels = {}
    legend_items = defaultdict(list)
    for m_idx, plotters in enumerate(to_plot):
        for var_name, selection, isel, values in plotters:
            label = labeller.make_label_vert(var_name, selection, isel)
            data_label = data_labels[m_idx] if data_labels else None
            plotted = _d_helper(values.flatten(), label, colors[m_idx], bw, circular, line_width, markersize, hdi_prob, point_estimate, hdi_markers, outline, shade, axis_map[label])
            if data_label is not None:
                legend_items[axis_map[label]].append((data_label, plotted))
    for ax1, legend in legend_items.items():
        legend = Legend(items=legend, location='center_right', orientation='horizontal')
        ax1.add_layout(legend, 'above')
        ax1.legend.click_policy = 'hide'
    show_layout(ax, show)
    return ax