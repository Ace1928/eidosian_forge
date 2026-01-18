import numpy as np
from bokeh.models import CategoricalColorMapper, LinearColorMapper
from holoviews.core.dimension import Dimension
from holoviews.core.options import Cycle
from holoviews.core.spaces import HoloMap
from holoviews.element import Labels
from holoviews.plotting.bokeh.util import property_to_dict
from ..utils import ParamLogStream
from .test_plot import TestBokehPlot, bokeh_renderer
def test_label_categorical_color_op(self):
    labels = Labels([(0, 0, 'A'), (0, 1, 'B'), (0, 2, 'C')], vdims='color').opts(text_color='color')
    plot = bokeh_renderer.get_plot(labels)
    cds = plot.handles['cds']
    glyph = plot.handles['glyph']
    cmapper = plot.handles['text_color_color_mapper']
    self.assertTrue(cmapper, CategoricalColorMapper)
    self.assertEqual(cmapper.factors, ['A', 'B', 'C'])
    self.assertEqual(cds.data['text_color'], np.array(['A', 'B', 'C']))
    self.assertEqual(property_to_dict(glyph.text_color), {'field': 'text_color', 'transform': cmapper})