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
def encode_store_data(store_data):
    """
    Encode store_data dict into a JSON serializable dict

    This is currently done by pickling store_data and converting to a base64 encoded
    string. If HoloViews supports JSON serialization in the future, this method could
    be updated to use this approach instead

    Args:
        store_data: dict potentially containing HoloViews objects

    Returns:
        dict that can be JSON serialized
    """
    return {'pickled': base64.b64encode(pickle.dumps(store_data)).decode('utf-8')}