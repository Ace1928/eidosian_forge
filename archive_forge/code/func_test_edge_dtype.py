import numpy as np
from numpy.lib.histograms import histogram, histogramdd, histogram_bin_edges
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
import pytest
def test_edge_dtype(self):
    """ Test that if an edge array is input, its type is preserved """
    x = np.array([0, 10, 20])
    y = x / 10
    x_edges = np.array([0, 5, 15, 20])
    y_edges = x_edges / 10
    hist, edges = histogramdd((x, y), bins=(x_edges, y_edges))
    assert_equal(edges[0].dtype, x_edges.dtype)
    assert_equal(edges[1].dtype, y_edges.dtype)