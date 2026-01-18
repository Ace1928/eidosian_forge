import numpy as np
from holoviews.core.dimension import Dimension
from holoviews.core.options import Cycle
from holoviews.core.spaces import HoloMap
from holoviews.element import Labels
from holoviews.plotting.util import rgb2hex
from .test_plot import TestMPLPlot, mpl_renderer
def test_label_rotation_op(self):
    labels = Labels([(0, 0, 90), (0, 1, 180), (0, 2, 270)], vdims='rotation').opts(rotation='rotation')
    plot = mpl_renderer.get_plot(labels)
    artist = plot.handles['artist']
    self.assertEqual([a.get_rotation() for a in artist], [90, 180, 270])