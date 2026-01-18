import numpy as np
from holoviews.core import DynamicMap, HoloMap, NdOverlay, Overlay
from holoviews.element import Curve, Scatter
from ...utils import LoggingComparisonTestCase
from .test_plot import TestMPLPlot, mpl_renderer
def test_dynamic_subplot_remapping(self):

    def cb(X):
        return NdOverlay({i: Curve(np.arange(10) + i) for i in range(X - 2, X)})
    dmap = DynamicMap(cb, kdims=['X']).redim.range(X=(1, 10))
    plot = mpl_renderer.get_plot(dmap)
    plot.update((3,))
    for i, subplot in enumerate(plot.subplots.values()):
        self.assertEqual(subplot.cyclic_index, i + 3)
        self.assertEqual(list(subplot.overlay_dims.values()), [i + 1])