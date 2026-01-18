import numpy as np
from numpy.lib.histograms import histogram, histogramdd, histogram_bin_edges
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
import pytest
def test_last_bin_inclusive_range(self):
    arr = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 4.0, 5.0])
    hist, edges = np.histogram(arr, bins=30, range=(-0.5, 5))
    assert_equal(hist[-1], 1)