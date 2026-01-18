import numpy as np
from numpy.lib.histograms import histogram, histogramdd, histogram_bin_edges
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
import pytest
def test_density_non_uniform_2d(self):
    x_edges = np.array([0, 2, 8])
    y_edges = np.array([0, 6, 8])
    relative_areas = np.array([[3, 9], [1, 3]])
    x = np.array([1] + [1] * 3 + [7] * 3 + [7] * 9)
    y = np.array([7] + [1] * 3 + [7] * 3 + [1] * 9)
    hist, edges = histogramdd((y, x), bins=(y_edges, x_edges))
    assert_equal(hist, relative_areas)
    hist, edges = histogramdd((y, x), bins=(y_edges, x_edges), density=True)
    assert_equal(hist, 1 / (8 * 8))