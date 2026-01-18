import pytest
from mpmath import *
from mpmath.calculus.optimization import Secant, Muller, Bisection, Illinois, \
def test_bisection():
    assert findroot(lambda x: x ** 2 - 1, (0, 2), solver='bisect') == 1