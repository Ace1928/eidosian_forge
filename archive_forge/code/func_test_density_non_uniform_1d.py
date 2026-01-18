import numpy as np
from numpy.lib.histograms import histogram, histogramdd, histogram_bin_edges
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
import pytest
def test_density_non_uniform_1d(self):
    v = np.arange(10)
    bins = np.array([0, 1, 3, 6, 10])
    hist, edges = histogram(v, bins, density=True)
    hist_dd, edges_dd = histogramdd((v,), (bins,), density=True)
    assert_equal(hist, hist_dd)
    assert_equal(edges, edges_dd[0])