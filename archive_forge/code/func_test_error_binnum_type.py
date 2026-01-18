import numpy as np
from numpy.lib.histograms import histogram, histogramdd, histogram_bin_edges
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
import pytest
def test_error_binnum_type(self):
    vals = np.linspace(0.0, 1.0, num=100)
    histogram(vals, 5)
    assert_raises(TypeError, histogram, vals, 2.4)