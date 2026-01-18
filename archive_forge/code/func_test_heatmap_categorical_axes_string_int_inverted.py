import numpy as np
from bokeh.models import FactorRange, HoverTool, Range1d
from holoviews.element import HeatMap, Image, Points
from .test_plot import TestBokehPlot, bokeh_renderer
def test_heatmap_categorical_axes_string_int_inverted(self):
    hmap = HeatMap([('A', 1, 1), ('B', 2, 2)]).opts(invert_axes=True)
    plot = bokeh_renderer.get_plot(hmap)
    x_range = plot.handles['x_range']
    y_range = plot.handles['y_range']
    self.assertIsInstance(x_range, Range1d)
    self.assertEqual(x_range.start, 0.5)
    self.assertEqual(x_range.end, 2.5)
    self.assertIsInstance(y_range, FactorRange)
    self.assertEqual(y_range.factors, ['A', 'B'])