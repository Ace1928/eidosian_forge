import numpy as np
from holoviews.core.dimension import Dimension
from holoviews.core.options import Cycle
from holoviews.core.spaces import HoloMap
from holoviews.element import Labels
from holoviews.plotting.util import rgb2hex
from .test_plot import TestMPLPlot, mpl_renderer
def test_label_rotation_op_update(self):
    labels = HoloMap({0: Labels([(0, 0, 45), (0, 1, 180), (0, 2, 90)], vdims='rotation'), 1: Labels([(0, 0, 30), (0, 1, 120), (0, 2, 60)], vdims='rotation')}).opts(rotation='rotation')
    plot = mpl_renderer.get_plot(labels)
    artist = plot.handles['artist']
    self.assertEqual([a.get_rotation() for a in artist], [45, 180, 90])
    plot.update((1,))
    artist = plot.handles['artist']
    self.assertEqual([a.get_rotation() for a in artist], [30, 120, 60])