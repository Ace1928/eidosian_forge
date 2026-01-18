import numpy as np
from holoviews.element import (
from .test_plot import TestPlotlyPlot
def test_hline_layout(self):
    layout = (HLine(1) + HLine(2) + HLine(3) + HLine(4)).cols(2).opts(vspacing=0, hspacing=0)
    state = self._get_plot_state(layout)
    shapes = state['layout']['shapes']
    self.assertEqual(len(shapes), 4)
    self.assert_hline(shapes[0], 3, yref='y', xdomain=[0.0, 0.5])
    self.assert_hline(shapes[1], 4, yref='y2', xdomain=[0.5, 1.0])
    self.assert_hline(shapes[2], 1, yref='y3', xdomain=[0.0, 0.5])
    self.assert_hline(shapes[3], 2, yref='y4', xdomain=[0.5, 1.0])