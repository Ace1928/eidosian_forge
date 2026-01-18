from numpy.testing import (assert_equal, assert_almost_equal,
from scipy.fftpack import (diff, fft, ifft, tilbert, itilbert, hilbert,
import numpy as np
from numpy import arange, sin, cos, pi, exp, tanh, sum, sign
from numpy.random import random
def direct_tilbert(x, h=1, period=None):
    fx = fft(x)
    n = len(fx)
    if period is None:
        period = 2 * pi
    w = fftfreq(n) * h * 2 * pi / period * n
    w[0] = 1
    w = 1j / tanh(w)
    w[0] = 0j
    return ifft(w * fx)