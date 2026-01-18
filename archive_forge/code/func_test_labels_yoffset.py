import numpy as np
from holoviews.core.options import Cycle
from holoviews.core.spaces import HoloMap
from holoviews.element import Labels, Tiles
from .test_plot import TestPlotlyPlot
def test_labels_yoffset(self):
    offset = 20000
    labels = Tiles('') * Labels([(self.xs[0], self.ys[0], 'A'), (self.xs[1], self.ys[1], 'B'), (self.xs[2], self.ys[2], 'C')]).opts(yoffset=offset)
    state = self._get_plot_state(labels)
    lons, lats = Tiles.easting_northing_to_lon_lat(self.xs, np.array(self.ys) + offset)
    self.assertEqual(state['data'][1]['lon'], lons)
    self.assertEqual(state['data'][1]['lat'], lats)