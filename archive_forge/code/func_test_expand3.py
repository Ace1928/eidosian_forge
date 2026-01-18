from symengine.test_utilities import raises
from symengine import (Symbol, Integer, Add, Mul, Pow, Rational, sqrt,
def test_expand3():
    x = Symbol('x')
    y = Symbol('y')
    assert ((1 / (x * y) - x * y + 2) * (1 + x * y)).expand() == 3 + 1 / (x * y) + x * y - (x * y) ** 2