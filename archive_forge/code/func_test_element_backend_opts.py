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
def test_element_backend_opts(self):
    heat_map = HeatMap([(1, 2, 3), (2, 3, 4), (3, 4, 5)]).opts(colorbar=True, backend_opts={'colorbar.title': 'Testing', 'colorbar.ticker': FixedTicker(ticks=(3.5, 5)), 'colorbar.major_label_overrides': {3.5: 'A', 5: 'B'}})
    plot = bokeh_renderer.get_plot(heat_map)
    colorbar = plot.handles['colorbar']
    self.assertEqual(colorbar.title, 'Testing')
    self.assertEqual(colorbar.ticker.ticks, (3.5, 5))
    self.assertEqual(colorbar.major_label_overrides, {3.5: 'A', 5: 'B'})