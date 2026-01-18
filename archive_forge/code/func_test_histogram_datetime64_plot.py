import datetime as dt
import numpy as np
from bokeh.models import CategoricalColorMapper, DatetimeAxis, LinearColorMapper
from holoviews.core.overlay import NdOverlay
from holoviews.element import Dataset, Histogram, Image, Points
from holoviews.operation import histogram
from holoviews.plotting.bokeh.util import property_to_dict
from ...utils import LoggingComparisonTestCase
from .test_plot import TestBokehPlot, bokeh_renderer
def test_histogram_datetime64_plot(self):
    dates = np.array([dt.datetime(2017, 1, i) for i in range(1, 5)])
    hist = histogram(Dataset(dates, 'Date'), num_bins=4)
    plot = bokeh_renderer.get_plot(hist)
    source = plot.handles['source']
    data = {'top': np.array([1, 1, 1, 1]), 'left': np.array(['2017-01-01T00:00:00.000000', '2017-01-01T18:00:00.000000', '2017-01-02T12:00:00.000000', '2017-01-03T06:00:00.000000'], dtype='datetime64[us]'), 'right': np.array(['2017-01-01T18:00:00.000000', '2017-01-02T12:00:00.000000', '2017-01-03T06:00:00.000000', '2017-01-04T00:00:00.000000'], dtype='datetime64[us]')}
    for k, v in data.items():
        self.assertEqual(source.data[k], v)
    xaxis = plot.handles['xaxis']
    range_x = plot.handles['x_range']
    self.assertIsInstance(xaxis, DatetimeAxis)
    self.assertEqual(range_x.start, np.datetime64('2017-01-01T00:00:00.000000', 'us'))
    self.assertEqual(range_x.end, np.datetime64('2017-01-04T00:00:00.000000', 'us'))