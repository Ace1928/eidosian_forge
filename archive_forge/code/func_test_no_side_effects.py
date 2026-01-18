import numpy as np
from numpy.lib.histograms import histogram, histogramdd, histogram_bin_edges
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
import pytest
def test_no_side_effects(self):
    values = np.array([1.3, 2.5, 2.3])
    np.histogram(values, range=[-10, 10], bins=100)
    assert_array_almost_equal(values, [1.3, 2.5, 2.3])