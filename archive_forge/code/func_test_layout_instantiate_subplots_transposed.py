import numpy as np
from holoviews.element import Curve, Image
from .test_plot import TestPlotlyPlot, plotly_renderer
def test_layout_instantiate_subplots_transposed(self):
    layout = Curve(range(10)) + Curve(range(10)) + Image(np.random.rand(10, 10)) + Curve(range(10)) + Curve(range(10))
    plot = plotly_renderer.get_plot(layout.opts(transpose=True))
    positions = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)]
    self.assertEqual(sorted(plot.subplots.keys()), positions)