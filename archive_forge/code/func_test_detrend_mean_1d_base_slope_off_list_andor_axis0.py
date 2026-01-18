from numpy.testing import (assert_allclose, assert_almost_equal,
import numpy as np
import pytest
from matplotlib import mlab
def test_detrend_mean_1d_base_slope_off_list_andor_axis0(self):
    input = self.sig_base + self.sig_slope + self.sig_off
    target = self.sig_base + self.sig_slope_mean
    self.allclose(mlab.detrend_mean(input, axis=0), target)
    self.allclose(mlab.detrend_mean(input.tolist()), target)
    self.allclose(mlab.detrend_mean(input.tolist(), axis=0), target)