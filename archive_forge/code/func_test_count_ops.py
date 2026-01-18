from symengine.test_utilities import raises
from symengine import (Symbol, Integer, Add, Mul, Pow, Rational, sqrt,
def test_count_ops():
    x, y = symbols('x, y')
    assert count_ops(x + y) == 1
    assert count_ops((x + y, x * y)) == 2
    assert count_ops([[x ** y], [x + y - 1]]) == 3
    assert count_ops(x + y, x * y) == 2