import numpy as np
from holoviews.element import (
from .test_plot import TestPlotlyPlot
def assert_vline(self, shape, x, xref='x', ydomain=(0, 1)):
    self.assertEqual(shape['type'], 'line')
    self.assertEqual(shape['x0'], x)
    self.assertEqual(shape['x1'], x)
    self.assertEqual(shape['xref'], xref)
    self.assertEqual(shape['y0'], ydomain[0])
    self.assertEqual(shape['y1'], ydomain[1])
    self.assertEqual(shape['yref'], 'paper')