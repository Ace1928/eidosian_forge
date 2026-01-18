import numpy as np
from bokeh.models import CategoricalColorMapper, LinearAxis, LinearColorMapper
from holoviews.core.overlay import NdOverlay, Overlay
from holoviews.element import Bars
from holoviews.plotting.bokeh.util import property_to_dict
from ..utils import ParamLogStream
from .test_plot import TestBokehPlot, bokeh_renderer
def test_bars_fill_alpha_op(self):
    bars = Bars([(0, 0, 0), (0, 1, 0.2), (0, 2, 0.7)], vdims=['y', 'alpha']).opts(fill_alpha='alpha')
    plot = bokeh_renderer.get_plot(bars)
    cds = plot.handles['cds']
    glyph = plot.handles['glyph']
    self.assertEqual(cds.data['fill_alpha'], np.array([0, 0.2, 0.7]))
    self.assertNotEqual(property_to_dict(glyph.line_alpha), {'field': 'fill_alpha'})
    self.assertEqual(property_to_dict(glyph.fill_alpha), {'field': 'fill_alpha'})