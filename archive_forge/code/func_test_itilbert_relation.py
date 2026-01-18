from numpy.testing import (assert_equal, assert_almost_equal,
from scipy.fftpack import (diff, fft, ifft, tilbert, itilbert, hilbert,
import numpy as np
from numpy import arange, sin, cos, pi, exp, tanh, sum, sign
from numpy.random import random
def test_itilbert_relation(self):
    for n in [16, 17, 64, 127]:
        x = arange(n) * 2 * pi / n
        f = sin(x) + cos(2 * x) * sin(x)
        y = ihilbert(f)
        y1 = direct_ihilbert(f)
        assert_array_almost_equal(y, y1)
        y2 = itilbert(f, h=10)
        assert_array_almost_equal(y, y2)