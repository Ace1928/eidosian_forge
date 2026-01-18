from sympy.core.numbers import (I, Rational, oo)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.calculus.singularities import (
from sympy.sets import Interval, FiniteSet
from sympy.testing.pytest import raises
from sympy.abc import x, y
def test_issue_23401():
    x = Symbol('x')
    expr = (x + 1) / (-0.001 * x ** 2 + 0.1 * x + 0.1)
    assert is_increasing(expr, Interval(1, 2), x)