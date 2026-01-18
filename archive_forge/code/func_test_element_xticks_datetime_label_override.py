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
def test_element_xticks_datetime_label_override(self):
    dates = [(dt.datetime(2016, 1, i), i) for i in range(1, 4)]
    tick = dt.datetime(2016, 1, 1, 12)
    curve = Curve(dates).opts(xticks=[(tick, 'A')])
    plot = bokeh_renderer.get_plot(curve)
    self.assertEqual(plot.state.xaxis.ticker.ticks, [dt_to_int(tick, 'ms')])
    self.assertEqual(plot.state.xaxis.major_label_overrides, {dt_to_int(tick, 'ms'): 'A'})