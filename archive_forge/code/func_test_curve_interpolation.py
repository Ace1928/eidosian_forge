import numpy as np
from holoviews.element import Curve, Tiles
from .test_plot import TestPlotlyPlot
def test_curve_interpolation(self):
    from holoviews.operation import interpolate_curve
    interp_xs = np.array([0.0, 0.5, 0.5, 1.5, 1.5, 2.0])
    interp_curve = interpolate_curve(Curve(self.ys), interpolation='steps-mid')
    interp_ys = interp_curve.dimension_values('y')
    interp_lons, interp_lats = Tiles.easting_northing_to_lon_lat(interp_xs, interp_ys)
    curve = Tiles('') * Curve(self.ys).opts(interpolation='steps-mid')
    state = self._get_plot_state(curve)
    self.assertEqual(state['data'][1]['lat'], interp_lats)
    self.assertEqual(state['data'][1]['lon'], interp_lons)