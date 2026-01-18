from symengine.test_utilities import raises
from symengine import (Symbol, Integer, Add, Mul, Pow, Rational, sqrt,
def test_expand1():
    x = Symbol('x')
    y = Symbol('y')
    z = Symbol('z')
    assert ((2 * x + y) ** 2).expand() == 4 * x ** 2 + 4 * x * y + y ** 2
    assert (x ** 2) ** 3 == x ** 6
    assert ((2 * x ** 2 + 3 * y) ** 2).expand() == 4 * x ** 4 + 12 * x ** 2 * y + 9 * y ** 2
    assert ((2 * x / 3 + y / 4) ** 2).expand() == 4 * x ** 2 / 9 + x * y / 3 + y ** 2 / 16