import numpy as np
from holoviews.core.spaces import GridSpace
from holoviews.element import Curve, Scatter
from .test_plot import TestPlotlyPlot
def test_grid_state(self):
    grid = GridSpace({(i, j): Curve([i, j]) for i in [0, 1] for j in [0, 1]})
    state = self._get_plot_state(grid)
    self.assertEqual(state['data'][0]['y'], np.array([0, 0]))
    self.assertEqual(state['data'][0]['xaxis'], 'x')
    self.assertEqual(state['data'][0]['yaxis'], 'y')
    self.assertEqual(state['data'][1]['y'], np.array([1, 0]))
    self.assertEqual(state['data'][1]['xaxis'], 'x2')
    self.assertEqual(state['data'][1]['yaxis'], 'y')
    self.assertEqual(state['data'][2]['y'], np.array([0, 1]))
    self.assertEqual(state['data'][2]['xaxis'], 'x')
    self.assertEqual(state['data'][2]['yaxis'], 'y2')
    self.assertEqual(state['data'][3]['y'], np.array([1, 1]))
    self.assertEqual(state['data'][3]['xaxis'], 'x2')
    self.assertEqual(state['data'][3]['yaxis'], 'y2')