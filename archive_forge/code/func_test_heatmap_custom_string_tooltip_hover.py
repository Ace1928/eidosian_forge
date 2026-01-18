import numpy as np
from bokeh.models import FactorRange, HoverTool, Range1d
from holoviews.element import HeatMap, Image, Points
from .test_plot import TestBokehPlot, bokeh_renderer
def test_heatmap_custom_string_tooltip_hover(self):
    tooltips = '<div><h1>Test</h1></div>'
    custom_hover = HoverTool(tooltips=tooltips)
    hm = HeatMap([(1, 1, 1), (2, 2, 0)], kdims=['x with space', 'y with $pecial symbol'])
    hm = hm.opts(tools=[custom_hover])
    plot = bokeh_renderer.get_plot(hm)
    hover = plot.handles['hover']
    self.assertEqual(hover.tooltips, tooltips)
    self.assertEqual(hover.renderers, [plot.handles['glyph_renderer']])