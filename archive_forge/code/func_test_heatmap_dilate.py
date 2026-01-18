import numpy as np
from bokeh.models import FactorRange, HoverTool, Range1d
from holoviews.element import HeatMap, Image, Points
from .test_plot import TestBokehPlot, bokeh_renderer
def test_heatmap_dilate(self):
    hmap = HeatMap([('A', 1, 1), ('B', 2, 2)]).opts(dilate=True)
    plot = bokeh_renderer.get_plot(hmap)
    glyph = plot.handles['glyph']
    self.assertTrue(glyph.dilate)