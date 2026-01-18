import datetime as dt
import numpy as np
from bokeh.models import CategoricalColorMapper, DatetimeAxis, LinearColorMapper
from holoviews.core.overlay import NdOverlay
from holoviews.element import Dataset, Histogram, Image, Points
from holoviews.operation import histogram
from holoviews.plotting.bokeh.util import property_to_dict
from ...utils import LoggingComparisonTestCase
from .test_plot import TestBokehPlot, bokeh_renderer
def test_histogram_padding_datetime_nonsquare(self):
    histogram = Histogram([(np.datetime64('2016-04-0%d' % i, 'ns'), i) for i in range(1, 4)]).opts(padding=0.1, width=600)
    plot = bokeh_renderer.get_plot(histogram)
    x_range, y_range = (plot.handles['x_range'], plot.handles['y_range'])
    self.assertEqual(x_range.start, np.datetime64('2016-03-31T08:24:00.000000000'))
    self.assertEqual(x_range.end, np.datetime64('2016-04-03T15:36:00.000000000'))
    self.assertEqual(y_range.start, 0)
    self.assertEqual(y_range.end, 3.2)