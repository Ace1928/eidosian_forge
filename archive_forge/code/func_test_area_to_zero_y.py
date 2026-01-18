import numpy as np
import pandas as pd
from holoviews.element import Area, Overlay
from .test_plot import TestPlotlyPlot
def test_area_to_zero_y(self):
    curve = Area([1, 2, 3])
    state = self._get_plot_state(curve)
    self.assertEqual(state['data'][0]['x'], np.array([0, 1, 2]))
    self.assertEqual(state['data'][0]['y'], np.array([1, 2, 3]))
    self.assertEqual(state['data'][0]['mode'], 'lines')
    self.assertEqual(state['data'][0]['fill'], 'tozeroy')
    self.assertEqual(state['layout']['yaxis']['range'], [0, 3])