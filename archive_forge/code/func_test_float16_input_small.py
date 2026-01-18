from numpy.testing import (assert_, assert_equal, assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft._pocketfft import (ifft, fft, fftn, ifftn,
from numpy import (arange, array, asarray, zeros, dot, exp, pi,
import numpy as np
import numpy.fft
from numpy.random import rand
@pytest.mark.parametrize('size', SMALL_COMPOSITE_SIZES + SMALL_PRIME_SIZES)
def test_float16_input_small(self, size):
    x = np.random.rand(size, size) + 1j * np.random.rand(size, size)
    y1 = fftn(x.real.astype(np.float16))
    y2 = fftn(x.real.astype(np.float64)).astype(np.complex64)
    assert_equal(y1.dtype, np.complex64)
    assert_array_almost_equal_nulp(y1, y2, 500000.0)