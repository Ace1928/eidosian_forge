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
def test_dynamic_subplot_creation(self):

    def cb(X):
        return NdOverlay({i: Curve(np.arange(10) + i) for i in range(X)})
    dmap = DynamicMap(cb, kdims=['X']).redim.range(X=(1, 10))
    plot = bokeh_renderer.get_plot(dmap)
    self.assertEqual(len(plot.subplots), 1)
    plot.update((3,))
    self.assertEqual(len(plot.subplots), 3)
    for i, subplot in enumerate(plot.subplots.values()):
        self.assertEqual(subplot.cyclic_index, i)