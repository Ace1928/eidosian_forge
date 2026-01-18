import numpy as np
from bokeh.models import CategoricalColorMapper, LinearColorMapper
from holoviews.element import ErrorBars
from holoviews.plotting.bokeh.util import property_to_dict
from .test_plot import TestBokehPlot, bokeh_renderer
def test_errorbars_line_width_op(self):
    errorbars = ErrorBars([(0, 0, 0.1, 0.2, 1), (0, 1, 0.2, 0.4, 4), (0, 2, 0.6, 1.2, 8)], vdims=['y', 'perr', 'nerr', 'line_width']).opts(line_width='line_width')
    plot = bokeh_renderer.get_plot(errorbars)
    cds = plot.handles['cds']
    glyph = plot.handles['glyph']
    self.assertEqual(cds.data['line_width'], np.array([1, 4, 8]))
    self.assertEqual(property_to_dict(glyph.line_width), {'field': 'line_width'})