import numpy as np
from holoviews.core.options import Cycle
from holoviews.core.spaces import HoloMap
from holoviews.element import Labels, Tiles
from .test_plot import TestPlotlyPlot
def test_labels_state(self):
    labels = Tiles('') * Labels([(self.xs[0], self.ys[0], 'A'), (self.xs[1], self.ys[1], 'B'), (self.xs[2], self.ys[2], 'C')]).redim.range(x=self.x_range, y=self.y_range)
    state = self._get_plot_state(labels)
    self.assertEqual(state['data'][1]['lon'], self.lons)
    self.assertEqual(state['data'][1]['lat'], self.lats)
    self.assertEqual(state['data'][1]['text'], ['A', 'B', 'C'])
    self.assertEqual(state['data'][1]['mode'], 'text')
    self.assertEqual(state['layout']['mapbox']['center'], {'lat': self.lat_center, 'lon': self.lon_center})