import base64
import copy
import pickle
import uuid
from collections import namedtuple
from dash.exceptions import PreventUpdate
import holoviews as hv
from holoviews.core.decollate import (
from holoviews.plotting.plotly import DynamicMap, PlotlyRenderer
from holoviews.plotting.plotly.callbacks import (
from holoviews.plotting.plotly.util import clean_internal_figure_properties
from holoviews.streams import Derived, History
import plotly.graph_objects as go
from dash import callback_context
from dash.dependencies import Input, Output, State
def plot_to_figure(plot, reset_nclicks=0, layout_ranges=None, responsive=True, use_ranges=True):
    """
    Convert a HoloViews plotly plot to a plotly.py Figure.

    Args:
        plot: A HoloViews plotly plot object
        reset_nclicks: Number of times a reset button associated with the plot has been
            clicked

    Returns:
        A plotly.py Figure
    """
    fig_dict = plot.state
    clean_internal_figure_properties(fig_dict)
    fig_dict['layout']['uirevision'] = 'reset-' + str(reset_nclicks)
    if layout_ranges and use_ranges:
        for k in fig_dict['layout']:
            if k.startswith(('xaxis', 'yaxis')):
                fig_dict['layout'][k].pop('range', None)
            if k.startswith('mapbox'):
                fig_dict['layout'][k].pop('zoom', None)
                fig_dict['layout'][k].pop('center', None)
    if responsive:
        fig_dict['layout'].pop('autosize', None)
    if responsive is True or responsive == 'width':
        fig_dict['layout'].pop('width', None)
    if responsive is True or responsive == 'height':
        fig_dict['layout'].pop('height', None)
    fig = go.Figure(fig_dict)
    if layout_ranges and use_ranges:
        fig.update_layout(layout_ranges)
    return fig