from sympy.core.numbers import (I, Rational, oo)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.calculus.singularities import (
from sympy.sets import Interval, FiniteSet
from sympy.testing.pytest import raises
from sympy.abc import x, y
def test_is_monotonic():
    """Test whether is_monotonic returns correct value."""
    assert is_monotonic(1 / (x ** 2 - 3 * x), Interval.open(Rational(3, 2), 3))
    assert is_monotonic(1 / (x ** 2 - 3 * x), Interval.open(1.5, 3))
    assert is_monotonic(1 / (x ** 2 - 3 * x), Interval.Lopen(3, oo))
    assert is_monotonic(x ** 3 - 3 * x ** 2 + 4 * x, S.Reals)
    assert not is_monotonic(-x ** 2, S.Reals)
    assert is_monotonic(x ** 2 + y + 1, Interval(1, 2), x)
    raises(NotImplementedError, lambda: is_monotonic(x ** 2 + y + 1))