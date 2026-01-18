from numpy.testing import (assert_allclose, assert_almost_equal,
import numpy as np
import pytest
from matplotlib import mlab
def test_detrend_mean_2d(self):
    input = np.vstack([self.sig_off, self.sig_base + self.sig_off])
    target = np.vstack([self.sig_zeros, self.sig_base])
    self.allclose(mlab.detrend_mean(input), target)
    self.allclose(mlab.detrend_mean(input, axis=None), target)
    self.allclose(mlab.detrend_mean(input.T, axis=None).T, target)
    self.allclose(mlab.detrend(input), target)
    self.allclose(mlab.detrend(input, axis=None), target)
    self.allclose(mlab.detrend(input.T, key='constant', axis=None), target.T)
    input = np.vstack([self.sig_base, self.sig_base + self.sig_off, self.sig_base + self.sig_slope, self.sig_base + self.sig_off + self.sig_slope])
    target = np.vstack([self.sig_base, self.sig_base, self.sig_base + self.sig_slope_mean, self.sig_base + self.sig_slope_mean])
    self.allclose(mlab.detrend_mean(input.T, axis=0), target.T)
    self.allclose(mlab.detrend_mean(input, axis=1), target)
    self.allclose(mlab.detrend_mean(input, axis=-1), target)
    self.allclose(mlab.detrend(input, key='default', axis=1), target)
    self.allclose(mlab.detrend(input.T, key='mean', axis=0), target.T)
    self.allclose(mlab.detrend(input.T, key=mlab.detrend_mean, axis=0), target.T)