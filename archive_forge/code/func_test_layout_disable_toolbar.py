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
def test_layout_disable_toolbar(self):
    layout = (Curve([]) + Points([])).opts(toolbar=None)
    plot = bokeh_renderer.get_plot(layout)
    self.assertIsInstance(plot.state, GridPlot)
    self.assertEqual(len(plot.state.children), 2)