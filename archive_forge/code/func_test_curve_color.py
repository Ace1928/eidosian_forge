import numpy as np
from holoviews.element import Curve, Tiles
from .test_plot import TestPlotlyPlot
def test_curve_color(self):
    curve = Tiles('') * Curve([1, 2, 3]).opts(color='red')
    state = self._get_plot_state(curve)
    self.assertEqual(state['data'][1]['line']['color'], 'red')