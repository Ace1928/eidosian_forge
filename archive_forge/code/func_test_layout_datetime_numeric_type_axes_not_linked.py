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
def test_layout_datetime_numeric_type_axes_not_linked(self):
    curve1 = Curve([1, 2, 3])
    curve2 = Curve([(dt.datetime(2020, 1, 1), 0), (dt.datetime(2020, 1, 2), 1), (dt.datetime(2020, 1, 3), 2)])
    layout = curve1 + curve2
    plot = bokeh_renderer.get_plot(layout)
    cp1, cp2 = (plot.subplots[0, 0].subplots['main'], plot.subplots[0, 1].subplots['main'])
    self.assertIsNot(cp1.handles['x_range'], cp2.handles['x_range'])
    self.assertIs(cp1.handles['y_range'], cp2.handles['y_range'])