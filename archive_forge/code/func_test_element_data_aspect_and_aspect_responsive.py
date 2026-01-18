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
def test_element_data_aspect_and_aspect_responsive(self):
    curve = Curve([0, 2]).opts(data_aspect=1, aspect=2, responsive=True)
    plot = bokeh_renderer.get_plot(curve)
    self.assertEqual(plot.state.aspect_ratio, 0.5)
    self.assertEqual(plot.state.aspect_scale, 1)
    self.assertEqual(plot.state.sizing_mode, 'scale_both')
    x_range = plot.handles['x_range']
    y_range = plot.handles['y_range']
    self.assertEqual(x_range.start, 0)
    self.assertEqual(x_range.end, 1)
    self.assertEqual(y_range.start, 0)
    self.assertEqual(y_range.end, 2)