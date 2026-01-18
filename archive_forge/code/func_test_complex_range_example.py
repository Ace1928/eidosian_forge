import numpy as np
import panel as pn
from bokeh.models import FactorRange, FixedTicker, HoverTool, Range1d, Span
from holoviews.core import DynamicMap, HoloMap, NdOverlay, Overlay
from holoviews.element import (
from holoviews.plotting.bokeh.util import property_to_dict
from holoviews.streams import Stream, Tap
from holoviews.util import Dynamic
from ...utils import LoggingComparisonTestCase
from .test_plot import TestBokehPlot, bokeh_renderer
def test_complex_range_example(self):
    errors = [(0.1 * i, np.sin(0.1 * i), (i + 1) / 3.0, (i + 1) / 5.0) for i in np.linspace(0, 100, 11)]
    errors = ErrorBars(errors, vdims=['y', 'yerrneg', 'yerrpos']).redim.range(y=(0, None))
    overlay = Curve(errors) * errors * VLine(4)
    plot = bokeh_renderer.get_plot(overlay)
    x_range = plot.handles['x_range']
    y_range = plot.handles['y_range']
    self.assertEqual(x_range.start, 0)
    self.assertEqual(x_range.end, 10.0)
    self.assertEqual(y_range.start, 0)
    self.assertEqual(y_range.end, 19.655978889110628)