from symengine.test_utilities import raises
from symengine import (Symbol, Integer, Add, Mul, Pow, Rational, sqrt,
def test_arit4():
    x = Symbol('x')
    y = Symbol('y')
    assert x * x == x ** 2
    assert x * y == y * x
    assert x * x * x == x ** 3
    assert x * y * x * x == x ** 3 * y