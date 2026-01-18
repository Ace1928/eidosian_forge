import numpy as np
from numpy.lib.histograms import histogram, histogramdd, histogram_bin_edges
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
import pytest
def test_identical_samples(self):
    x = np.zeros((10, 2), int)
    hist, edges = histogramdd(x, bins=2)
    assert_array_equal(edges[0], np.array([-0.5, 0.0, 0.5]))