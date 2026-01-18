from sympy.core.numbers import (I, Rational, oo)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.calculus.singularities import (
from sympy.sets import Interval, FiniteSet
from sympy.testing.pytest import raises
from sympy.abc import x, y
def test_is_increasing():
    """Test whether is_increasing returns correct value."""
    a = Symbol('a', negative=True)
    assert is_increasing(x ** 3 - 3 * x ** 2 + 4 * x, S.Reals)
    assert is_increasing(-x ** 2, Interval(-oo, 0))
    assert not is_increasing(-x ** 2, Interval(0, oo))
    assert not is_increasing(4 * x ** 3 - 6 * x ** 2 - 72 * x + 30, Interval(-2, 3))
    assert is_increasing(x ** 2 + y, Interval(1, oo), x)
    assert is_increasing(-x ** 2 * a, Interval(1, oo), x)
    assert is_increasing(1)
    assert is_increasing(4 * x ** 3 - 6 * x ** 2 - 72 * x + 30, Interval(-2, 3)) is False