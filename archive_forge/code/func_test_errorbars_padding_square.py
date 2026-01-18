import numpy as np
from bokeh.models import CategoricalColorMapper, LinearColorMapper
from holoviews.element import ErrorBars
from holoviews.plotting.bokeh.util import property_to_dict
from .test_plot import TestBokehPlot, bokeh_renderer
def test_errorbars_padding_square(self):
    errorbars = ErrorBars([(1, 1, 0.5), (2, 2, 0.5), (3, 3, 0.5)]).opts(padding=0.1)
    plot = bokeh_renderer.get_plot(errorbars)
    x_range, y_range = (plot.handles['x_range'], plot.handles['y_range'])
    self.assertEqual(x_range.start, 0.8)
    self.assertEqual(x_range.end, 3.2)
    self.assertEqual(y_range.start, 0.19999999999999996)
    self.assertEqual(y_range.end, 3.8)