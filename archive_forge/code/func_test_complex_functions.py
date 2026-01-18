from mpmath.libmp import *
from mpmath import *
import random
import time
import math
import cmath
def test_complex_functions():
    for x in list(range(10)) + list(range(-10, 0)):
        for y in list(range(10)) + list(range(-10, 0)):
            z = complex(x, y) / 4.3 + 0.01j
            assert exp(mpc(z)).ae(cmath.exp(z))
            assert log(mpc(z)).ae(cmath.log(z))
            assert cos(mpc(z)).ae(cmath.cos(z))
            assert sin(mpc(z)).ae(cmath.sin(z))
            assert tan(mpc(z)).ae(cmath.tan(z))
            assert sinh(mpc(z)).ae(cmath.sinh(z))
            assert cosh(mpc(z)).ae(cmath.cosh(z))
            assert tanh(mpc(z)).ae(cmath.tanh(z))