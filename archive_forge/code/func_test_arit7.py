from symengine.test_utilities import raises
from symengine import (Symbol, Integer, Add, Mul, Pow, Rational, sqrt,
def test_arit7():
    x = Symbol('x')
    y = Symbol('y')
    assert x - x == 0
    assert x - y != y - x
    assert 2 * x - x == x
    assert 3 * x - x == 2 * x
    assert 2 * x * y - x * y == x * y