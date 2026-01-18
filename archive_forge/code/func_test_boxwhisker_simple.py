import numpy as np
from holoviews.element import BoxWhisker
from .test_plot import TestMPLPlot, mpl_renderer
def test_boxwhisker_simple(self):
    values = np.random.rand(100)
    boxwhisker = BoxWhisker(values)
    plot = mpl_renderer.get_plot(boxwhisker)
    data, style, axis_opts = plot.get_data(boxwhisker, {}, {})
    self.assertEqual(data[0][0], values)
    self.assertEqual(style['labels'], [''])