from numpy.testing import (assert_allclose, assert_almost_equal,
import numpy as np
import pytest
from matplotlib import mlab
def test_detrend_mean(self):
    for sig in [0.0, 5.5]:
        assert mlab.detrend_mean(sig) == 0.0
        assert mlab.detrend(sig, key='mean') == 0.0
        assert mlab.detrend(sig, key=mlab.detrend_mean) == 0.0
    self.allclose(mlab.detrend_mean(self.sig_zeros), self.sig_zeros)
    self.allclose(mlab.detrend_mean(self.sig_base), self.sig_base)
    self.allclose(mlab.detrend_mean(self.sig_base + self.sig_off), self.sig_base)
    self.allclose(mlab.detrend_mean(self.sig_base + self.sig_slope), self.sig_base + self.sig_slope_mean)
    self.allclose(mlab.detrend_mean(self.sig_base + self.sig_slope + self.sig_off), self.sig_base + self.sig_slope_mean)