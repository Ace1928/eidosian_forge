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
def get_layout_ranges(plot):
    layout_ranges = {}
    fig_dict = plot.state
    for k in fig_dict['layout']:
        if k.startswith(('xaxis', 'yaxis')):
            if 'range' in fig_dict['layout'][k]:
                layout_ranges[k] = {'range': fig_dict['layout'][k]['range']}
        if k.startswith('mapbox'):
            mapbox_ranges = {}
            if 'center' in fig_dict['layout'][k]:
                mapbox_ranges['center'] = fig_dict['layout'][k]['center']
            if 'zoom' in fig_dict['layout'][k]:
                mapbox_ranges['zoom'] = fig_dict['layout'][k]['zoom']
            if mapbox_ranges:
                layout_ranges[k] = mapbox_ranges
    return layout_ranges