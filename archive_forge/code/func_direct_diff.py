from numpy.testing import (assert_equal, assert_almost_equal,
from scipy.fftpack import (diff, fft, ifft, tilbert, itilbert, hilbert,
import numpy as np
from numpy import arange, sin, cos, pi, exp, tanh, sum, sign
from numpy.random import random
def direct_diff(x, k=1, period=None):
    fx = fft(x)
    n = len(fx)
    if period is None:
        period = 2 * pi
    w = fftfreq(n) * 2j * pi / period * n
    if k < 0:
        w = 1 / w ** k
        w[0] = 0.0
    else:
        w = w ** k
    if n > 2000:
        w[250:n - 250] = 0.0
    return ifft(w * fx).real