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
def test_layout_shared_inverted_yaxis(self):
    layout = (Curve([]) + Curve([])).opts('Curve', invert_yaxis=True)
    plot = bokeh_renderer.get_plot(layout)
    subplot = next(iter(plot.subplots.values())).subplots['main']
    self.assertEqual(subplot.handles['y_range'].start, 1)
    self.assertEqual(subplot.handles['y_range'].end, 0)