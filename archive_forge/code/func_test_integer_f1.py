import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal,
from pytest import raises as assert_raises
import scipy.signal._waveforms as waveforms
def test_integer_f1(self):
    f0 = 10.0
    t1 = 3.0
    t = np.linspace(-1, 1, 11)
    f1 = 20.0
    float_result = waveforms.chirp(t, f0, t1, f1)
    f1 = 20
    int_result = waveforms.chirp(t, f0, t1, f1)
    err_msg = "Integer input 'f1=20' gives wrong result"
    assert_equal(int_result, float_result, err_msg=err_msg)