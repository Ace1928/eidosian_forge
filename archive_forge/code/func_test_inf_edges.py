import numpy as np
from numpy.lib.histograms import histogram, histogramdd, histogram_bin_edges
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
import pytest
def test_inf_edges(self):
    with np.errstate(invalid='ignore'):
        x = np.arange(6).reshape(3, 2)
        expected = np.array([[1, 0], [0, 1], [0, 1]])
        h, e = np.histogramdd(x, bins=[3, [-np.inf, 2, 10]])
        assert_allclose(h, expected)
        h, e = np.histogramdd(x, bins=[3, np.array([-1, 2, np.inf])])
        assert_allclose(h, expected)
        h, e = np.histogramdd(x, bins=[3, [-np.inf, 3, np.inf]])
        assert_allclose(h, expected)