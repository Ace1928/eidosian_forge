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
def test_dimensioned_streams_with_dynamic_callback_returns_layout(self):
    stream = Stream.define('aname', aname='a')()

    def cb(aname):
        x = np.linspace(0, 1, 10)
        y = np.random.randn(10)
        curve = Curve((x, y), group=aname)
        hist = Histogram(y)
        return (curve + hist).opts(shared_axes=False)
    m = DynamicMap(cb, kdims=['aname'], streams=[stream])
    p = bokeh_renderer.get_plot(m)
    T = 'XYZT'
    stream.event(aname=T)
    self.assertIn('aname: ' + T, p.handles['title'].text, p.handles['title'].text)
    p.cleanup()
    self.assertEqual(stream._subscribers, [])