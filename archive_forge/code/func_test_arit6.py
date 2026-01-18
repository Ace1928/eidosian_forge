from symengine.test_utilities import raises
from symengine import (Symbol, Integer, Add, Mul, Pow, Rational, sqrt,
def test_arit6():
    x = Symbol('x')
    y = Symbol('y')
    e = x + y
    assert str(e) == 'x + y' or 'y + x'
    e = x * y
    assert str(e) == 'x*y' or 'y*x'
    e = Integer(2) * x
    assert str(e) == '2*x'
    e = 2 * x
    assert str(e) == '2*x'