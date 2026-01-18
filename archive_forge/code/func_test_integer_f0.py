import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal,
from pytest import raises as assert_raises
import scipy.signal._waveforms as waveforms
def test_integer_f0(self):
    f1 = 20.0
    t1 = 3.0
    t = np.linspace(-1, 1, 11)
    f0 = 10.0
    float_result = waveforms.chirp(t, f0, t1, f1)
    f0 = 10
    int_result = waveforms.chirp(t, f0, t1, f1)
    err_msg = "Integer input 'f0=10' gives wrong result"
    assert_equal(int_result, float_result, err_msg=err_msg)