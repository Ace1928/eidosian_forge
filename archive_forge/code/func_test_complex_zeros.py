import mpmath
from mpmath import *
from mpmath.libmp import *
import random
import sys
def test_complex_zeros():
    for a in [0, 2]:
        for b in [0, 3]:
            for c in [0, 4]:
                for d in [0, 5]:
                    assert mpc(a, b) * mpc(c, d) == complex(a, b) * complex(c, d)