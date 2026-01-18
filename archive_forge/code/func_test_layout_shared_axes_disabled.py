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
def test_layout_shared_axes_disabled(self):
    layout = (Curve([1, 2, 3]) + Curve([10, 20, 30])).opts(shared_axes=False)
    plot = bokeh_renderer.get_plot(layout)
    cp1, cp2 = (plot.subplots[0, 0].subplots['main'], plot.subplots[0, 1].subplots['main'])
    self.assertFalse(cp1.handles['y_range'] is cp2.handles['y_range'])
    self.assertEqual(cp1.handles['y_range'].start, 1)
    self.assertEqual(cp1.handles['y_range'].end, 3)
    self.assertEqual(cp2.handles['y_range'].start, 10)
    self.assertEqual(cp2.handles['y_range'].end, 30)