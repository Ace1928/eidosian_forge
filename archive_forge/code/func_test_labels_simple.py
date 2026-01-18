import numpy as np
from bokeh.models import CategoricalColorMapper, LinearColorMapper
from holoviews.core.dimension import Dimension
from holoviews.core.options import Cycle
from holoviews.core.spaces import HoloMap
from holoviews.element import Labels
from holoviews.plotting.bokeh.util import property_to_dict
from ..utils import ParamLogStream
from .test_plot import TestBokehPlot, bokeh_renderer
def test_labels_simple(self):
    labels = Labels([(0, 1, 'A'), (1, 0, 'B')])
    plot = bokeh_renderer.get_plot(labels)
    source = plot.handles['source']
    glyph = plot.handles['glyph']
    expected = {'x': np.array([0, 1]), 'y': np.array([1, 0]), 'Label': ['A', 'B']}
    for k, vals in expected.items():
        self.assertEqual(source.data[k], vals)
    self.assertEqual(glyph.x, 'x')
    self.assertEqual(glyph.y, 'y')
    self.assertEqual(glyph.text, 'Label')