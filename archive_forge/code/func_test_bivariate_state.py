import numpy as np
from holoviews.element import Bivariate
from .test_plot import TestPlotlyPlot
def test_bivariate_state(self):
    bivariate = Bivariate(([3, 2, 1], [0, 1, 2]))
    state = self._get_plot_state(bivariate)
    self.assertEqual(state['data'][0]['type'], 'histogram2dcontour')
    self.assertEqual(state['data'][0]['x'], np.array([3, 2, 1]))
    self.assertEqual(state['data'][0]['y'], np.array([0, 1, 2]))
    self.assertEqual(state['layout']['xaxis']['range'], [1, 3])
    self.assertEqual(state['layout']['yaxis']['range'], [0, 2])
    self.assertEqual(state['data'][0]['contours']['coloring'], 'lines')