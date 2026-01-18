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
def test_layout_update_visible(self):
    hmap = HoloMap({i: Curve(np.arange(i), label='A') for i in range(1, 3)})
    hmap2 = HoloMap({i: Curve(np.arange(i), label='B') for i in range(3, 5)})
    plot = bokeh_renderer.get_plot(hmap + hmap2)
    subplot1, subplot2 = (p for k, p in sorted(plot.subplots.items()))
    subplot1 = subplot1.subplots['main']
    subplot2 = subplot2.subplots['main']
    self.assertTrue(subplot1.handles['glyph_renderer'].visible)
    self.assertFalse(subplot2.handles['glyph_renderer'].visible)
    plot.update((4,))
    self.assertFalse(subplot1.handles['glyph_renderer'].visible)
    self.assertTrue(subplot2.handles['glyph_renderer'].visible)