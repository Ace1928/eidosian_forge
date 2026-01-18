import unittest
from unittest.mock import MagicMock, patch
from dash._callback_context import CallbackContext
from holoviews import Bounds, DynamicMap, Scatter
from holoviews.plotting.plotly.dash import (
from holoviews.streams import BoundsXY, RangeXY, Selection1D
from .test_plot import TestPlotlyPlot
import plotly.io as pio
def dmap_fn(x_range, y_range):
    x_range = (0, 1) if x_range is None else x_range
    y_range = (0, 1) if y_range is None else y_range
    return Scatter([[x_range[0], y_range[0]], [x_range[1], y_range[1]]], kdims=['x1'], vdims=['y1'])