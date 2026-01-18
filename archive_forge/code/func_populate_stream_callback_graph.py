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
def populate_stream_callback_graph(stream_callbacks, streams):
    """
    Populate the stream_callbacks dict with StreamCallback instances
    associated with all of the History and Derived streams in input stream list.

    Input streams to any History or Derived streams are processed recursively

    Args:
        stream_callbacks:  dict from id(stream) to StreamCallbacks the should
            be populated. Order will be a breadth-first traversal of the provided
            streams list, and any input streams that these depend on.

        streams: List of streams to build StreamCallbacks from

    Returns:
        None
    """
    for stream in streams:
        if isinstance(stream, Derived):
            cb = build_derived_callback(stream)
            if cb.output_id not in stream_callbacks:
                stream_callbacks[cb.output_id] = cb
                populate_stream_callback_graph(stream_callbacks, stream.input_streams)
        elif isinstance(stream, History):
            cb = build_history_callback(stream)
            if cb.output_id not in stream_callbacks:
                stream_callbacks[cb.output_id] = cb
                populate_stream_callback_graph(stream_callbacks, [stream.input_stream])