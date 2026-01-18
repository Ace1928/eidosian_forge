from sympy.core.numbers import (E, I, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import (Abs, re)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (cos, cot, csc, sec, sin, tan)
from sympy.functions.special.error_functions import expint
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.simplify.simplify import simplify
from sympy.calculus.util import (function_range, continuous_domain, not_empty_in,
from sympy.sets.sets import (Interval, FiniteSet, Complement, Union)
from sympy.testing.pytest import raises, _both_exp_pow
from sympy.abc import x
def test_continuous_domain():
    x = Symbol('x')
    assert continuous_domain(sin(x), x, Interval(0, 2 * pi)) == Interval(0, 2 * pi)
    assert continuous_domain(tan(x), x, Interval(0, 2 * pi)) == Union(Interval(0, pi / 2, False, True), Interval(pi / 2, pi * Rational(3, 2), True, True), Interval(pi * Rational(3, 2), 2 * pi, True, False))
    assert continuous_domain((x - 1) / (x - 1) ** 2, x, S.Reals) == Union(Interval(-oo, 1, True, True), Interval(1, oo, True, True))
    assert continuous_domain(log(x) + log(4 * x - 1), x, S.Reals) == Interval(Rational(1, 4), oo, True, True)
    assert continuous_domain(1 / sqrt(x - 3), x, S.Reals) == Interval(3, oo, True, True)
    assert continuous_domain(1 / x - 2, x, S.Reals) == Union(Interval.open(-oo, 0), Interval.open(0, oo))
    assert continuous_domain(1 / (x ** 2 - 4) + 2, x, S.Reals) == Union(Interval.open(-oo, -2), Interval.open(-2, 2), Interval.open(2, oo))
    domain = continuous_domain(log(tan(x) ** 2 + 1), x, S.Reals)
    assert not domain.contains(3 * pi / 2)
    assert domain.contains(5)
    d = Symbol('d', even=True, zero=False)
    assert continuous_domain(x ** (1 / d), x, S.Reals) == Interval(0, oo)