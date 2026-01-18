import datetime as dt
from unittest import SkipTest
import numpy as np
import panel as pn
import pytest
from bokeh.document import Document
from bokeh.models import (
from holoviews.core import DynamicMap, HoloMap, NdOverlay
from holoviews.core.util import dt_to_int
from holoviews.element import Curve, HeatMap, Image, Labels, Scatter
from holoviews.plotting.util import process_cmap
from holoviews.streams import PointDraw, Stream
from holoviews.util import render
from ...utils import LoggingComparisonTestCase
from .test_plot import TestBokehPlot, bokeh_renderer
def test_element_grid_custom_yticker(self):
    curve = Curve([1, 2, 3]).opts(yticks=[0.5, 2.5], show_grid=True)
    plot = bokeh_renderer.get_plot(curve)
    self.assertIs(plot.state.ygrid[0].ticker, plot.state.yaxis[0].ticker)