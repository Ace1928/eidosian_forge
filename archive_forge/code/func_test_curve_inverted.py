import numpy as np
from holoviews.element import Curve, Tiles
from .test_plot import TestPlotlyPlot
def test_curve_inverted(self):
    curve = Tiles('') * Curve([1, 2, 3]).opts(invert_axes=True)
    with self.assertRaises(ValueError) as e:
        self._get_plot_state(curve)
    self.assertIn('invert_axes', str(e.exception))