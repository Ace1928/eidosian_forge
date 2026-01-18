import unittest
import numpy as np
import holoviews as hv
from holoviews.element import (
from holoviews.plotting.bokeh.util import bokeh32, bokeh33
from .test_plot import TestBokehPlot, bokeh_renderer
def test_hspan_empty(self):
    vline = HSpan(None)
    plot = bokeh_renderer.get_plot(vline)
    span = plot.handles['glyph']
    self.assertEqual(span.visible, False)