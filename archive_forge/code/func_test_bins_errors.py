import numpy as np
from numpy.lib.histograms import histogram, histogramdd, histogram_bin_edges
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
import pytest
def test_bins_errors(self):
    x = np.arange(8).reshape(2, 4)
    assert_raises(ValueError, np.histogramdd, x, bins=[-1, 2, 4, 5])
    assert_raises(ValueError, np.histogramdd, x, bins=[1, 0.99, 1, 1])
    assert_raises(ValueError, np.histogramdd, x, bins=[1, 1, 1, [1, 2, 3, -3]])
    assert_(np.histogramdd(x, bins=[1, 1, 1, [1, 2, 3, 4]]))