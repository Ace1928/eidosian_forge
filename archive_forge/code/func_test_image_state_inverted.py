import numpy as np
from holoviews.element import Image
from .test_plot import TestPlotlyPlot
def test_image_state_inverted(self):
    img = Image(([1, 2, 3], [0, 1], np.array([[0, 1, 2], [2, 3, 4]]))).opts(invert_axes=True)
    state = self._get_plot_state(img)
    self.assertEqual(state['data'][0]['y0'], 1)
    self.assertEqual(state['data'][0]['dy'], 1)
    self.assertEqual(state['data'][0]['x0'], 0)
    self.assertEqual(state['data'][0]['dx'], 1)
    self.assertEqual(state['data'][0]['z'], np.array([[0, 1, 2], [2, 3, 4]]).T)
    self.assertEqual(state['data'][0]['zmin'], 0)
    self.assertEqual(state['data'][0]['zmax'], 4)
    self.assertEqual(state['layout']['yaxis']['range'], [0.5, 3.5])
    self.assertEqual(state['layout']['xaxis']['range'], [-0.5, 1.5])