from mpmath.libmp import *
from mpmath import *
import random
import time
import math
import cmath
def test_complex_powers():
    for dps in [15, 30, 100]:
        mp.dps = dps
        a = mpc(1j) ** 0.5
        assert a.real == a.imag == mpf(2) ** 0.5 / 2
    mp.dps = 15
    random.seed(1)
    for z1, z2 in random_complexes(100):
        assert (mpc(z1) ** mpc(z2)).ae(z1 ** z2, 1e-12)
    assert (e ** (-pi * 1j)).ae(-1)
    mp.dps = 50
    assert (e ** (-pi * 1j)).ae(-1)
    mp.dps = 15