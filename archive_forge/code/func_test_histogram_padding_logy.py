import datetime as dt
import numpy as np
from bokeh.models import CategoricalColorMapper, DatetimeAxis, LinearColorMapper
from holoviews.core.overlay import NdOverlay
from holoviews.element import Dataset, Histogram, Image, Points
from holoviews.operation import histogram
from holoviews.plotting.bokeh.util import property_to_dict
from ...utils import LoggingComparisonTestCase
from .test_plot import TestBokehPlot, bokeh_renderer
def test_histogram_padding_logy(self):
    histogram = Histogram([(1, 2), (2, 1), (3, 3)]).opts(padding=0.1, logy=True)
    plot = bokeh_renderer.get_plot(histogram)
    x_range, y_range = (plot.handles['x_range'], plot.handles['y_range'])
    self.assertEqual(x_range.start, 0.19999999999999996)
    self.assertEqual(x_range.end, 3.8)
    self.assertEqual(y_range.start, 0.01)
    self.assertEqual(y_range.end, 3.348369522101713)
    self.log_handler.assertContains('WARNING', 'Logarithmic axis range encountered value less than')