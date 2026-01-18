from mpmath.libmp import *
from mpmath import *
import random
import time
import math
import cmath
def test_float_cbrt():
    mp.dps = 30
    for a in arange(0, 10, 0.1):
        assert cbrt(a * a * a).ae(a, eps)
    assert cbrt(-1).ae(0.5 + j * sqrt(3) / 2)
    one_third = mpf(1) / 3
    for a in arange(0, 10, 2.7) + [0.1 + 10 ** 5]:
        a = mpc(a + 1.1j)
        r1 = cbrt(a)
        mp.dps += 10
        r2 = pow(a, one_third)
        mp.dps -= 10
        assert r1.ae(r2, eps)
    mp.dps = 100
    for n in range(100, 301, 100):
        w = 10 ** n + j * 10 ** (-3)
        z = w * w * w
        r = cbrt(z)
        assert mpc_ae(r, w, eps)
    mp.dps = 15