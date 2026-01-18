import numpy as np
from holoviews.core.spaces import GridSpace
from holoviews.element import Curve, Scatter
from .test_plot import TestPlotlyPlot
def test_layout_with_grid(self):
    grid = GridSpace({(i, j): Curve([i, j]) for i in [0, 1] for j in [0, 1]})
    grid = grid.opts(vspacing=0, hspacing=0)
    scatter = Scatter([-10, 0])
    layout = (scatter + grid).opts(vspacing=0, hspacing=0)
    state = self._get_plot_state(layout)
    start_fig_width = 400
    grid_fig_width = 400 * 1.1
    x_domain_break1 = start_fig_width / (start_fig_width + grid_fig_width)
    start_fig_height = 400
    grid_fig_height = start_fig_height * 1.1
    y_domain_break1 = start_fig_height / grid_fig_height
    self.assertEqual(state['data'][0]['y'], np.array([-10, 0]))
    self.assertEqual(state['data'][0]['mode'], 'markers')
    self.assertEqual(state['data'][0]['xaxis'], 'x')
    self.assertEqual(state['data'][0]['yaxis'], 'y')
    self.assertEqual(state['layout']['xaxis']['range'], [0, 1])
    self.assertEqual(state['layout']['xaxis']['domain'], [0, x_domain_break1])
    self.assertEqual(state['layout']['yaxis']['range'], [-10, 1])
    self.assertEqual(state['layout']['yaxis']['domain'], [0, y_domain_break1])
    self.assertEqual(state['data'][1]['y'], np.array([0, 0]))
    self.assertEqual(state['data'][1]['mode'], 'lines')
    self.assertEqual(state['data'][1]['xaxis'], 'x2')
    self.assertEqual(state['data'][1]['yaxis'], 'y2')
    self.assertEqual(state['data'][2]['y'], np.array([1, 0]))
    self.assertEqual(state['data'][2]['mode'], 'lines')
    self.assertEqual(state['data'][2]['xaxis'], 'x3')
    self.assertEqual(state['data'][2]['yaxis'], 'y2')
    self.assertEqual(state['data'][3]['y'], np.array([0, 1]))
    self.assertEqual(state['data'][3]['mode'], 'lines')
    self.assertEqual(state['data'][3]['xaxis'], 'x2')
    self.assertEqual(state['data'][3]['yaxis'], 'y3')
    self.assertEqual(state['data'][4]['y'], np.array([1, 1]))
    self.assertEqual(state['data'][4]['mode'], 'lines')
    self.assertEqual(state['data'][4]['xaxis'], 'x3')
    self.assertEqual(state['data'][4]['yaxis'], 'y3')
    x_dimain_break2 = x_domain_break1 + (1 - x_domain_break1) / 2
    self.assertEqual(state['layout']['xaxis2']['domain'], [x_domain_break1, x_dimain_break2])
    self.assertEqual(state['layout']['xaxis3']['domain'], [x_dimain_break2, 1.0])
    self.assertEqual(state['layout']['yaxis2']['domain'], [0, 0.5])
    self.assertEqual(state['layout']['yaxis3']['domain'], [0.5, 1.0])