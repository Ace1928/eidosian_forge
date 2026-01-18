import numpy as np
import pytest
from holoviews.core.options import AbbreviatedException
from holoviews.core.spaces import HoloMap
from holoviews.element import ErrorBars
from .test_plot import TestMPLPlot, mpl_renderer
def test_errorbars_line_width_op_update(self):
    errorbars = HoloMap({0: ErrorBars([(0, 0, 0.1, 0.2, 1), (0, 1, 0.2, 0.4, 4), (0, 2, 0.6, 1.2, 8)], vdims=['y', 'perr', 'nerr', 'line_width']), 1: ErrorBars([(0, 0, 0.1, 0.2, 2), (0, 1, 0.2, 0.4, 6), (0, 2, 0.6, 1.2, 4)], vdims=['y', 'perr', 'nerr', 'line_width'])}).opts(linewidth='line_width')
    plot = mpl_renderer.get_plot(errorbars)
    artist = plot.handles['artist']
    self.assertEqual(artist.get_linewidths(), [1, 4, 8])
    plot.update((1,))
    self.assertEqual(artist.get_linewidths(), [2, 6, 4])