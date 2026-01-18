import numpy as np
from holoviews.element import Bars
from .test_plot import TestPlotlyPlot
def test_bars_stacked(self):
    bars = Bars([('A', 1, 1), ('B', 2, 2), ('C', 2, 3), ('C', 1, 4)], kdims=['A', 'B']).opts(stacked=True)
    state = self._get_plot_state(bars)
    self.assertEqual(state['data'][0]['x'], ['A', 'B', 'C'])
    self.assertEqual(state['data'][0]['y'], np.array([0, 2, 3]))
    self.assertEqual(state['data'][0]['type'], 'bar')
    self.assertEqual(state['data'][1]['x'], ['A', 'B', 'C'])
    self.assertEqual(state['data'][1]['y'], np.array([1, 0, 4]))
    self.assertEqual(state['data'][1]['type'], 'bar')
    self.assertEqual(state['layout']['barmode'], 'stack')
    self.assertEqual(state['layout']['xaxis']['range'], [None, None])
    self.assertEqual(state['layout']['xaxis']['title']['text'], 'A')
    self.assertEqual(state['layout']['yaxis']['range'], [0, 7.6])
    self.assertEqual(state['layout']['yaxis']['title']['text'], 'y')