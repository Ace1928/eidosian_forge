from mpmath.libmp import *
from mpmath import *
import random
import time
import math
import cmath
def test_complex_inverse_functions():
    mp.dps = 15
    iv.dps = 15
    for z1, z2 in random_complexes(30):
        assert sinh(asinh(z1)).ae(z1)
        assert acosh(z1).ae(cmath.acosh(z1))
        assert atanh(z1).ae(cmath.atanh(z1))
        assert atan(z1).ae(cmath.atan(z1))
        assert asin(z1).ae(cmath.asin(z1), rel_eps=1e-12)
        assert acos(z1).ae(cmath.acos(z1), rel_eps=1e-12)
        one = mpf(1)
    for i in range(-9, 10, 3):
        for k in range(-9, 10, 3):
            a = 0.9 * j * 10 ** k + 0.8 * one * 10 ** i
            b = cos(acos(a))
            assert b.ae(a)
            b = sin(asin(a))
            assert b.ae(a)
    one = mpf(1)
    err = 2 * 10 ** (-15)
    for i in range(-9, 9, 3):
        for k in range(-9, 9, 3):
            a = -0.9 * 10 ** k + j * 0.8 * one * 10 ** i
            b = cosh(acosh(a))
            assert b.ae(a, err)
            b = sinh(asinh(a))
            assert b.ae(a, err)