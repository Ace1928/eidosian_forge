from numpy.testing import (assert_equal, assert_almost_equal,
from scipy.fftpack import (diff, fft, ifft, tilbert, itilbert, hilbert,
import numpy as np
from numpy import arange, sin, cos, pi, exp, tanh, sum, sign
from numpy.random import random
def direct_hilbert(x):
    fx = fft(x)
    n = len(fx)
    w = fftfreq(n) * n
    w = 1j * sign(w)
    return ifft(w * fx)