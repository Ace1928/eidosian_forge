import numpy as np
from holoviews.element import Curve, Tiles
from .test_plot import TestPlotlyPlot
def test_curve_line_width(self):
    curve = Tiles('') * Curve([1, 2, 3]).opts(line_width=5)
    state = self._get_plot_state(curve)
    self.assertEqual(state['data'][1]['line']['width'], 5)