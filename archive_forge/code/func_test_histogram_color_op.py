import datetime as dt
import numpy as np
from bokeh.models import CategoricalColorMapper, DatetimeAxis, LinearColorMapper
from holoviews.core.overlay import NdOverlay
from holoviews.element import Dataset, Histogram, Image, Points
from holoviews.operation import histogram
from holoviews.plotting.bokeh.util import property_to_dict
from ...utils import LoggingComparisonTestCase
from .test_plot import TestBokehPlot, bokeh_renderer
def test_histogram_color_op(self):
    histogram = Histogram([(0, 0, '#000'), (0, 1, '#F00'), (0, 2, '#0F0')], vdims=['y', 'color']).opts(color='color')
    plot = bokeh_renderer.get_plot(histogram)
    cds = plot.handles['cds']
    glyph = plot.handles['glyph']
    self.assertEqual(cds.data['color'], np.array(['#000', '#F00', '#0F0']))
    self.assertEqual(property_to_dict(glyph.fill_color), {'field': 'color'})
    self.assertEqual(glyph.line_color, 'black')