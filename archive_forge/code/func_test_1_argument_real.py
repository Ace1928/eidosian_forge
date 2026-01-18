from numpy.testing import (assert_, assert_equal, assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft._pocketfft import (ifft, fft, fftn, ifftn,
from numpy import (arange, array, asarray, zeros, dot, exp, pi,
import numpy as np
import numpy.fft
from numpy.random import rand
def test_1_argument_real(self):
    x1 = np.array([1, 2, 3, 4], dtype=np.float16)
    y = fft(x1, n=4)
    assert_equal(y.dtype, np.complex64)
    assert_equal(y.shape, (4,))
    assert_array_almost_equal(y, direct_dft(x1.astype(np.float32)))