from mpmath.libmp import *
from mpmath import *
import random
import time
import math
import cmath
def test_float_sqrt():
    mp.dps = 15
    for x in [0, 1e-07, 0.1, 0.5, 1, 2, 3, 4, 5, 0.333, 76.19]:
        assert sqrt(mpf(x)) == float(x) ** 0.5
    assert sqrt(-1) == 1j
    assert sqrt(-2).ae(cmath.sqrt(-2))
    assert sqrt(-3).ae(cmath.sqrt(-3))
    assert sqrt(-100).ae(cmath.sqrt(-100))
    assert sqrt(1j).ae(cmath.sqrt(1j))
    assert sqrt(-1j).ae(cmath.sqrt(-1j))
    assert sqrt(math.pi + math.e * 1j).ae(cmath.sqrt(math.pi + math.e * 1j))
    assert sqrt(math.pi - math.e * 1j).ae(cmath.sqrt(math.pi - math.e * 1j))