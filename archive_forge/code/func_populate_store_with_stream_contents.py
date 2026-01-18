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
def populate_store_with_stream_contents(store_data, streams):
    """
    Add contents of streams to the store dictionary

    Args:
        store_data: The store dictionary
        streams: List of streams whose contents should be added to the store

    Returns:
        None
    """
    for stream in streams:
        store_data['streams'][id(stream)] = copy.deepcopy(stream.contents)
        if isinstance(stream, Derived):
            populate_store_with_stream_contents(store_data, stream.input_streams)
        elif isinstance(stream, History):
            populate_store_with_stream_contents(store_data, [stream.input_stream])