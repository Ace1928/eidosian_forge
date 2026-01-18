import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal,
from pytest import raises as assert_raises
import scipy.signal._waveforms as waveforms
def test_hyperbolic_at_zero(self):
    w = waveforms.chirp(t=0, f0=10.0, f1=1.0, t1=1.0, method='hyperbolic')
    assert_almost_equal(w, 1.0)