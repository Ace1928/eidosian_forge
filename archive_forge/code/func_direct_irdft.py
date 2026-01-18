from numpy.testing import (assert_, assert_equal, assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft._pocketfft import (ifft, fft, fftn, ifftn,
from numpy import (arange, array, asarray, zeros, dot, exp, pi,
import numpy as np
import numpy.fft
from numpy.random import rand
def direct_irdft(x, n):
    x = asarray(x)
    x1 = zeros(n, dtype=cdouble)
    for i in range(n // 2 + 1):
        x1[i] = x[i]
        if i > 0 and 2 * i < n:
            x1[n - i] = np.conj(x[i])
    return direct_idft(x1).real