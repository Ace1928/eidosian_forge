import datetime as dt
import re
import numpy as np
from bokeh.models import Div, GlyphRenderer, GridPlot, Spacer, Tabs, Title, Toolbar
from bokeh.models.layouts import TabPanel
from bokeh.plotting import figure
from holoviews.core import (
from holoviews.element import Curve, Histogram, Image, Points, Scatter
from holoviews.streams import Stream
from holoviews.util import opts, render
from holoviews.util.transform import dim
from ...utils import LoggingComparisonTestCase
from .test_plot import TestBokehPlot, bokeh_renderer
def test_layout_empty_subplots(self):
    layout = Curve(range(10)) + NdOverlay() + HoloMap() + HoloMap({1: Image(np.random.rand(10, 10))})
    plot = bokeh_renderer.get_plot(layout)
    self.assertEqual(len(plot.subplots.values()), 2)
    self.log_handler.assertContains('WARNING', 'skipping subplot')
    self.log_handler.assertContains('WARNING', 'skipping subplot')