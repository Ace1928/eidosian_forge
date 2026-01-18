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
def test_colormapper_logz_float_zero_bound(self):
    img = Image(np.array([[0, 1], [2, 3.0]])).opts(logz=True)
    plot = bokeh_renderer.get_plot(img)
    cmapper = plot.handles['color_mapper']
    self.assertEqual(cmapper.low, 0)
    self.assertEqual(cmapper.high, 3)
    self.log_handler.assertContains('WARNING', 'Log color mapper lower bound <= 0')