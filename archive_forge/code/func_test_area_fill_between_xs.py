import numpy as np
import pandas as pd
from holoviews.element import Area, Overlay
from .test_plot import TestPlotlyPlot
def test_area_fill_between_xs(self):
    area = Area([(0, 1, 0.5), (1, 2, 1), (2, 3, 2.25)], vdims=['y', 'y2']).opts(invert_axes=True)
    state = self._get_plot_state(area)
    self.assertEqual(state['data'][0]['x'], np.array([0.5, 1, 2.25]))
    self.assertEqual(state['data'][0]['mode'], 'lines')
    self.assertEqual(state['data'][0].get('fill', None), None)
    self.assertEqual(state['data'][1]['x'], np.array([1, 2, 3]))
    self.assertEqual(state['data'][1]['mode'], 'lines')
    self.assertEqual(state['data'][1]['fill'], 'tonextx')
    self.assertEqual(state['layout']['xaxis']['range'], [0.5, 3])
    self.assertEqual(state['layout']['yaxis']['range'], [0, 2])