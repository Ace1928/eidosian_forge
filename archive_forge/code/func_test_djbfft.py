from numpy.testing import (assert_, assert_equal, assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft._pocketfft import (ifft, fft, fftn, ifftn,
from numpy import (arange, array, asarray, zeros, dot, exp, pi,
import numpy as np
import numpy.fft
from numpy.random import rand
def test_djbfft(self):
    for i in range(2, 14):
        n = 2 ** i
        x = np.arange(-1, n, 2) + 1j * np.arange(0, n + 1, 2)
        x[0] = 0
        if n % 2 == 0:
            x[-1] = np.real(x[-1])
        y1 = np.fft.irfft(x)
        y = irfft(x)
        assert_array_almost_equal(y, y1)