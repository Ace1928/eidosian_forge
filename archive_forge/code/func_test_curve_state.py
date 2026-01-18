import numpy as np
from holoviews.element import Curve, Tiles
from .test_plot import TestPlotlyPlot
def test_curve_state(self):
    curve = Tiles('') * Curve((self.xs, self.ys)).redim.range(x=self.x_range, y=self.y_range)
    state = self._get_plot_state(curve)
    self.assertEqual(state['data'][1]['lon'], self.lons)
    self.assertEqual(state['data'][1]['lat'], self.lats)
    self.assertEqual(state['data'][1]['mode'], 'lines')
    self.assertEqual(state['layout']['mapbox']['center'], {'lat': self.lat_center, 'lon': self.lon_center})