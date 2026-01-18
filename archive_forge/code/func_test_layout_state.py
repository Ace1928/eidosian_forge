import numpy as np
from holoviews.element import Curve, Image
from .test_plot import TestPlotlyPlot, plotly_renderer
def test_layout_state(self):
    layout = Curve([1, 2, 3]) + Curve([2, 4, 6])
    state = self._get_plot_state(layout)
    self.assertEqual(state['data'][0]['y'], np.array([1, 2, 3]))
    self.assertEqual(state['data'][0]['yaxis'], 'y')
    self.assertEqual(state['data'][1]['y'], np.array([2, 4, 6]))
    self.assertEqual(state['data'][1]['yaxis'], 'y2')