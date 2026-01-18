from symengine.test_utilities import raises
from symengine import (Symbol, Integer, Add, Mul, Pow, Rational, sqrt,
def test_from_args():
    x = Symbol('x')
    y = Symbol('y')
    assert Add._from_args([]) == 0
    assert Add._from_args([x]) == x
    assert Add._from_args([x, y]) == x + y
    assert Mul._from_args([]) == 1
    assert Mul._from_args([x]) == x
    assert Mul._from_args([x, y]) == x * y