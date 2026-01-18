from symengine.test_utilities import raises
from symengine import (Symbol, Integer, Add, Mul, Pow, Rational, sqrt,
def test_Pow_base_exp():
    x = Symbol('x')
    y = Symbol('y')
    e = Pow(x + y, 2)
    assert isinstance(e, Pow)
    assert e.exp == 2
    assert e.base == x + y
    assert sqrt(x - 1).as_base_exp() == (x - 1, Rational(1, 2))