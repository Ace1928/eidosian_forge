from symengine.test_utilities import raises
from symengine import (Symbol, Integer, Add, Mul, Pow, Rational, sqrt,
def test_arit8():
    x = Symbol('x')
    y = Symbol('y')
    z = Symbol('z')
    assert x ** y * x ** x == x ** (x + y)
    assert x ** y * x ** x * x ** z == x ** (x + y + z)
    assert x ** y - x ** y == 0
    assert x ** 2 / x == x
    assert y * x ** 2 / (x * y) == x
    assert (2 * x ** 3 * y ** 2 * z) ** 3 / 8 == x ** 9 * y ** 6 * z ** 3
    assert 2 * y ** (-2 * x ** 2) * (3 * y ** (2 * x ** 2)) == 6