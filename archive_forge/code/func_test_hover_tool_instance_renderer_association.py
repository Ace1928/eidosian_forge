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
def test_hover_tool_instance_renderer_association(self):
    tooltips = [('index', '$index')]
    hover = HoverTool(tooltips=tooltips)
    overlay = Curve(np.random.rand(10, 2)).opts(tools=[hover]) * Points(np.random.rand(10, 2))
    plot = bokeh_renderer.get_plot(overlay)
    curve_plot = plot.subplots['Curve', 'I']
    self.assertEqual(len(curve_plot.handles['hover'].renderers), 1)
    self.assertIn(curve_plot.handles['glyph_renderer'], curve_plot.handles['hover'].renderers)
    self.assertEqual(plot.handles['hover'].tooltips, tooltips)