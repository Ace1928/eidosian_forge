import pytest
from mpmath import *
from mpmath.calculus.optimization import Secant, Muller, Bisection, Illinois, \
def test_muller():
    f = lambda x: (2 + x) ** 3 + 2
    x = findroot(f, 1.0, solver=Muller)
    assert abs(f(x)) < eps