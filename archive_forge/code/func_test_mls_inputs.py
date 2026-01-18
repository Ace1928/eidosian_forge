import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from pytest import raises as assert_raises
from numpy.fft import fft, ifft
from scipy.signal import max_len_seq
def test_mls_inputs(self):
    assert_raises(ValueError, max_len_seq, 10, state=np.zeros(10))
    assert_raises(ValueError, max_len_seq, 10, state=np.ones(3))
    assert_raises(ValueError, max_len_seq, 10, length=-1)
    assert_array_equal(max_len_seq(10, length=0)[0], [])
    assert_raises(ValueError, max_len_seq, 64)
    assert_raises(ValueError, max_len_seq, 10, taps=[-1, 1])