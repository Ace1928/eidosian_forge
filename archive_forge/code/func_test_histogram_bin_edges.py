import numpy as np
from numpy.lib.histograms import histogram, histogramdd, histogram_bin_edges
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
import pytest
def test_histogram_bin_edges(self):
    hist, e = histogram([1, 2, 3, 4], [1, 2])
    edges = histogram_bin_edges([1, 2, 3, 4], [1, 2])
    assert_array_equal(edges, e)
    arr = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 4.0, 5.0])
    hist, e = histogram(arr, bins=30, range=(-0.5, 5))
    edges = histogram_bin_edges(arr, bins=30, range=(-0.5, 5))
    assert_array_equal(edges, e)
    hist, e = histogram(arr, bins='auto', range=(0, 1))
    edges = histogram_bin_edges(arr, bins='auto', range=(0, 1))
    assert_array_equal(edges, e)