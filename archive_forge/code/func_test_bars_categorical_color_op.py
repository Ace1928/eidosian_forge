import numpy as np
from bokeh.models import CategoricalColorMapper, LinearAxis, LinearColorMapper
from holoviews.core.overlay import NdOverlay, Overlay
from holoviews.element import Bars
from holoviews.plotting.bokeh.util import property_to_dict
from ..utils import ParamLogStream
from .test_plot import TestBokehPlot, bokeh_renderer
def test_bars_categorical_color_op(self):
    bars = Bars([(0, 0, 'A'), (0, 1, 'B'), (0, 2, 'C')], vdims=['y', 'color']).opts(color='color')
    plot = bokeh_renderer.get_plot(bars)
    cds = plot.handles['cds']
    glyph = plot.handles['glyph']
    cmapper = plot.handles['color_color_mapper']
    self.assertTrue(cmapper, CategoricalColorMapper)
    self.assertEqual(cmapper.factors, ['A', 'B', 'C'])
    self.assertEqual(cds.data['color'], np.array(['A', 'B', 'C']))
    self.assertEqual(property_to_dict(glyph.fill_color), {'field': 'color', 'transform': cmapper})
    self.assertEqual(property_to_dict(glyph.line_color), 'black')