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
def test_layout_plot_tabs_with_adjoints(self):
    layout = (Curve([]) + Curve([]).hist()).opts(tabs=True)
    plot = bokeh_renderer.get_plot(layout)
    self.assertIsInstance(plot.state, Tabs)
    panel1, panel2 = plot.state.tabs
    self.assertIsInstance(panel1, TabPanel)
    self.assertIsInstance(panel2, TabPanel)
    self.assertEqual(panel1.title, 'Curve I')
    self.assertEqual(panel2.title, 'AdjointLayout I')