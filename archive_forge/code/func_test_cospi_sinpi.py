from mpmath.libmp import *
from mpmath import *
import random
import time
import math
import cmath
def test_cospi_sinpi():
    assert sinpi(0) == 0
    assert sinpi(0.5) == 1
    assert sinpi(1) == 0
    assert sinpi(1.5) == -1
    assert sinpi(2) == 0
    assert sinpi(2.5) == 1
    assert sinpi(-0.5) == -1
    assert cospi(0) == 1
    assert cospi(0.5) == 0
    assert cospi(1) == -1
    assert cospi(1.5) == 0
    assert cospi(2) == 1
    assert cospi(2.5) == 0
    assert cospi(-0.5) == 0
    assert cospi(100000000000.25).ae(sqrt(2) / 2)
    a = cospi(2 + 3j)
    assert a.real.ae(cos((2 + 3j) * pi).real)
    assert a.imag == 0
    b = sinpi(2 + 3j)
    assert b.imag.ae(sin((2 + 3j) * pi).imag)
    assert b.real == 0
    mp.dps = 35
    x1 = mpf(10000) - mpf('1e-15')
    x2 = mpf(10000) + mpf('1e-15')
    x3 = mpf(10000.5) - mpf('1e-15')
    x4 = mpf(10000.5) + mpf('1e-15')
    x5 = mpf(10001) - mpf('1e-15')
    x6 = mpf(10001) + mpf('1e-15')
    x7 = mpf(10001.5) - mpf('1e-15')
    x8 = mpf(10001.5) + mpf('1e-15')
    mp.dps = 15
    M = 10 ** 15
    assert (sinpi(x1) * M).ae(-pi)
    assert (sinpi(x2) * M).ae(pi)
    assert (cospi(x3) * M).ae(pi)
    assert (cospi(x4) * M).ae(-pi)
    assert (sinpi(x5) * M).ae(pi)
    assert (sinpi(x6) * M).ae(-pi)
    assert (cospi(x7) * M).ae(-pi)
    assert (cospi(x8) * M).ae(pi)
    assert 0.999 < cospi(x1, rounding='d') < 1
    assert 0.999 < cospi(x2, rounding='d') < 1
    assert 0.999 < sinpi(x3, rounding='d') < 1
    assert 0.999 < sinpi(x4, rounding='d') < 1
    assert -1 < cospi(x5, rounding='d') < -0.999
    assert -1 < cospi(x6, rounding='d') < -0.999
    assert -1 < sinpi(x7, rounding='d') < -0.999
    assert -1 < sinpi(x8, rounding='d') < -0.999
    assert (sinpi(1e-15) * M).ae(pi)
    assert (sinpi(-1e-15) * M).ae(-pi)
    assert cospi(1e-15) == 1
    assert cospi(1e-15, rounding='d') < 1