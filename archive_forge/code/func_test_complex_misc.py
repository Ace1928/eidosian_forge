import mpmath
from mpmath import *
from mpmath.libmp import *
import random
import sys
def test_complex_misc():
    assert 1 + mpc(2) == 3
    assert not mpc(2).ae(2 + 1e-13)
    assert mpc(2 + 1e-15j).ae(2)