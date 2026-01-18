import numpy as np
from numpy.lib.histograms import histogram, histogramdd, histogram_bin_edges
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
import pytest
@pytest.mark.skip(reason='Bad memory reports lead to OOM in ci testing')
def test_big_arrays(self):
    sample = np.zeros([100000000, 3])
    xbins = 400
    ybins = 400
    zbins = np.arange(16000)
    hist = np.histogramdd(sample=sample, bins=(xbins, ybins, zbins))
    assert_equal(type(hist), type((1, 2)))