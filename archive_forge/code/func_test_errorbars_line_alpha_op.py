import numpy as np
from bokeh.models import CategoricalColorMapper, LinearColorMapper
from holoviews.element import ErrorBars
from holoviews.plotting.bokeh.util import property_to_dict
from .test_plot import TestBokehPlot, bokeh_renderer
def test_errorbars_line_alpha_op(self):
    errorbars = ErrorBars([(0, 0, 0.1, 0.2, 0), (0, 1, 0.2, 0.4, 0.2), (0, 2, 0.6, 1.2, 0.7)], vdims=['y', 'perr', 'nerr', 'alpha']).opts(line_alpha='alpha')
    plot = bokeh_renderer.get_plot(errorbars)
    cds = plot.handles['cds']
    glyph = plot.handles['glyph']
    self.assertEqual(cds.data['line_alpha'], np.array([0, 0.2, 0.7]))
    self.assertEqual(property_to_dict(glyph.line_alpha), {'field': 'line_alpha'})