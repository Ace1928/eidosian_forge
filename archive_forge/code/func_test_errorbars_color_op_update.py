import numpy as np
import pytest
from holoviews.core.options import AbbreviatedException
from holoviews.core.spaces import HoloMap
from holoviews.element import ErrorBars
from .test_plot import TestMPLPlot, mpl_renderer
def test_errorbars_color_op_update(self):
    errorbars = HoloMap({0: ErrorBars([(0, 0, 0.1, 0.2, '#000000'), (0, 1, 0.2, 0.4, '#FF0000'), (0, 2, 0.6, 1.2, '#00FF00')], vdims=['y', 'perr', 'nerr', 'color']), 1: ErrorBars([(0, 0, 0.1, 0.2, '#FF0000'), (0, 1, 0.2, 0.4, '#00FF00'), (0, 2, 0.6, 1.2, '#0000FF')], vdims=['y', 'perr', 'nerr', 'color'])}).opts(color='color')
    plot = mpl_renderer.get_plot(errorbars)
    artist = plot.handles['artist']
    self.assertEqual(artist.get_edgecolors(), np.array([[0, 0, 0, 1], [1, 0, 0, 1], [0, 1, 0, 1]]))
    plot.update((1,))
    self.assertEqual(artist.get_edgecolors(), np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]))