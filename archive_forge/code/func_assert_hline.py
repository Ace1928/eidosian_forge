import numpy as np
from holoviews.element import (
from .test_plot import TestPlotlyPlot
def assert_hline(self, shape, y, yref='y', xdomain=(0, 1)):
    self.assertEqual(shape['type'], 'line')
    self.assertEqual(shape['y0'], y)
    self.assertEqual(shape['y1'], y)
    self.assertEqual(shape['yref'], yref)
    self.assertEqual(shape['x0'], xdomain[0])
    self.assertEqual(shape['x1'], xdomain[1])
    self.assertEqual(shape['xref'], 'paper')