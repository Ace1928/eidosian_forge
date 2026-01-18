import numpy as np
from bokeh.models import CategoricalColorMapper, LinearColorMapper
from holoviews.core.dimension import Dimension
from holoviews.core.options import Cycle
from holoviews.core.spaces import HoloMap
from holoviews.element import Labels
from holoviews.plotting.bokeh.util import property_to_dict
from ..utils import ParamLogStream
from .test_plot import TestBokehPlot, bokeh_renderer
def test_label_angle_op(self):
    labels = Labels([(0, 0, 0), (0, 1, 45), (0, 2, 90)], vdims='angle').opts(angle='angle')
    plot = bokeh_renderer.get_plot(labels)
    cds = plot.handles['cds']
    glyph = plot.handles['glyph']
    self.assertEqual(cds.data['angle'], np.array([0, 0.785398, 1.570796]))
    self.assertEqual(property_to_dict(glyph.angle), {'field': 'angle'})