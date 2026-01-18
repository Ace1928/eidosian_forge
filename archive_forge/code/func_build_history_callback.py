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
def build_history_callback(history_stream):
    """
    Build StreamCallback for History stream

    Args:
        history_stream: A History stream

    Returns:
        StreamCallback
    """
    history_id = id(history_stream)
    input_stream_id = id(history_stream.input_stream)

    def history_callback(prior_value, input_value):
        new_value = copy.deepcopy(prior_value)
        new_value['values'].append(input_value)
        return new_value
    return StreamCallback(input_ids=[history_id, input_stream_id], fn=history_callback, output_id=history_id)