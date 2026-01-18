import numpy as np
from holoviews.core.dimension import Dimension
from holoviews.core.options import Cycle
from holoviews.core.spaces import HoloMap
from holoviews.element import Labels
from holoviews.plotting.util import rgb2hex
from .test_plot import TestMPLPlot, mpl_renderer
def test_label_color_op_update(self):
    labels = HoloMap({0: Labels([(0, 0, '#000000'), (0, 1, '#FF0000'), (0, 2, '#00FF00')], vdims='color'), 1: Labels([(0, 0, '#FF0000'), (0, 1, '#00FF00'), (0, 2, '#0000FF')], vdims='color')}).opts(color='color')
    plot = mpl_renderer.get_plot(labels)
    artist = plot.handles['artist']
    self.assertEqual([a.get_color() for a in artist], ['#000000', '#FF0000', '#00FF00'])
    plot.update((1,))
    artist = plot.handles['artist']
    self.assertEqual([a.get_color() for a in artist], ['#FF0000', '#00FF00', '#0000FF'])