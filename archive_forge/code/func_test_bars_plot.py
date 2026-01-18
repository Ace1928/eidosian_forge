import numpy as np
from holoviews.element import Bars
from .test_plot import TestPlotlyPlot
def test_bars_plot(self):
    bars = Bars([3, 2, 1])
    state = self._get_plot_state(bars)
    self.assertEqual(state['data'][0]['x'], ['0', '1', '2'])
    self.assertEqual(state['data'][0]['y'], np.array([3, 2, 1]))
    self.assertEqual(state['data'][0]['type'], 'bar')
    self.assertEqual(state['layout']['xaxis']['range'], [None, None])
    self.assertEqual(state['layout']['xaxis']['title']['text'], 'x')
    self.assertEqual(state['layout']['yaxis']['range'], [0, 3.2])
    self.assertEqual(state['layout']['yaxis']['title']['text'], 'y')