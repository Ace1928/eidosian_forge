from symengine.test_utilities import raises
from symengine import (Symbol, sin, cos, Integer, Add, I, RealDouble, ComplexDouble, sqrt)
from unittest.case import SkipTest
def test_eval_double2():
    x = Symbol('x')
    e = sin(x) ** 2 + sqrt(2)
    raises(RuntimeError, lambda: e.n(real=True))
    assert abs(e.n() - sin(x) ** 2.0 - 1.414) < 0.001