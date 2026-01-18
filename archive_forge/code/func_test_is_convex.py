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
def test_is_convex():
    assert is_convex(1 / x, x, domain=Interval.open(0, oo)) == True
    assert is_convex(1 / x, x, domain=Interval(-oo, 0)) == False
    assert is_convex(x ** 2, x, domain=Interval(0, oo)) == True
    assert is_convex(1 / x ** 3, x, domain=Interval.Lopen(0, oo)) == True
    assert is_convex(-1 / x ** 3, x, domain=Interval.Ropen(-oo, 0)) == True
    assert is_convex(log(x), x) == False
    raises(NotImplementedError, lambda: is_convex(log(x), x, a))