from sympy.core.numbers import (I, Rational, oo)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.calculus.singularities import (
from sympy.sets import Interval, FiniteSet
from sympy.testing.pytest import raises
from sympy.abc import x, y
def test_is_strictly_decreasing():
    """Test whether is_strictly_decreasing returns correct value."""
    assert is_strictly_decreasing(1 / (x ** 2 - 3 * x), Interval.Lopen(3, oo))
    assert not is_strictly_decreasing(1 / (x ** 2 - 3 * x), Interval.Ropen(-oo, Rational(3, 2)))
    assert not is_strictly_decreasing(-x ** 2, Interval(-oo, 0))
    assert not is_strictly_decreasing(1)
    assert is_strictly_decreasing(1 / (x ** 2 - 3 * x), Interval.open(Rational(3, 2), 3))
    assert is_strictly_decreasing(1 / (x ** 2 - 3 * x), Interval.open(1.5, 3))