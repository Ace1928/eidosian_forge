import mpmath
from mpmath import *
from mpmath.libmp import *
import random
import sys
def test_add_misc():
    mp.dps = 15
    assert mpf(4) + mpf(-70) == -66
    assert mpf(1) + mpf(1.1) / 80 == 1 + 1.1 / 80
    assert mpf((1, 10000000000)) + mpf(3) == mpf((1, 10000000000))
    assert mpf(3) + mpf((1, 10000000000)) == mpf((1, 10000000000))
    assert mpf((1, -10000000000)) + mpf(3) == mpf(3)
    assert mpf(3) + mpf((1, -10000000000)) == mpf(3)
    assert mpf(1) + 1e-15 != 1
    assert mpf(1) + 1e-20 == 1
    assert mpf(1.07e-22) + 0 == mpf(1.07e-22)
    assert mpf(0) + mpf(1.07e-22) == mpf(1.07e-22)