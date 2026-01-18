import datetime as dt
import numpy as np
from bokeh.models import CategoricalColorMapper, DatetimeAxis, LinearColorMapper
from holoviews.core.overlay import NdOverlay
from holoviews.element import Dataset, Histogram, Image, Points
from holoviews.operation import histogram
from holoviews.plotting.bokeh.util import property_to_dict
from ...utils import LoggingComparisonTestCase
from .test_plot import TestBokehPlot, bokeh_renderer
def test_histogram_line_width_op(self):
    histogram = Histogram([(0, 0, 1), (0, 1, 4), (0, 2, 8)], vdims=['y', 'line_width']).opts(line_width='line_width')
    plot = bokeh_renderer.get_plot(histogram)
    cds = plot.handles['cds']
    glyph = plot.handles['glyph']
    self.assertEqual(cds.data['line_width'], np.array([1, 4, 8]))
    self.assertEqual(property_to_dict(glyph.line_width), {'field': 'line_width'})