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
def build_derived_callback(derived_stream):
    """
    Build StreamCallback for Derived stream

    Args:
        derived_stream: A Derived stream

    Returns:
        StreamCallback
    """
    input_ids = [id(stream) for stream in derived_stream.input_streams]
    constants = copy.copy(derived_stream.constants)
    transform = derived_stream.transform_function

    def derived_callback(*stream_values):
        return transform(stream_values=stream_values, constants=constants)
    return StreamCallback(input_ids=input_ids, fn=derived_callback, output_id=id(derived_stream))