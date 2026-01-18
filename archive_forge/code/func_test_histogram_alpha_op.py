import datetime as dt
import numpy as np
from bokeh.models import CategoricalColorMapper, DatetimeAxis, LinearColorMapper
from holoviews.core.overlay import NdOverlay
from holoviews.element import Dataset, Histogram, Image, Points
from holoviews.operation import histogram
from holoviews.plotting.bokeh.util import property_to_dict
from ...utils import LoggingComparisonTestCase
from .test_plot import TestBokehPlot, bokeh_renderer
def test_histogram_alpha_op(self):
    histogram = Histogram([(0, 0, 0), (0, 1, 0.2), (0, 2, 0.7)], vdims=['y', 'alpha']).opts(alpha='alpha')
    plot = bokeh_renderer.get_plot(histogram)
    cds = plot.handles['cds']
    glyph = plot.handles['glyph']
    self.assertEqual(cds.data['alpha'], np.array([0, 0.2, 0.7]))
    self.assertEqual(property_to_dict(glyph.fill_alpha), {'field': 'alpha'})