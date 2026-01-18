from sympy.calculus.accumulationbounds import AccumBounds
from sympy.core.numbers import (E, Float, I, Rational, nan, oo, pi, zoo)
from sympy.core.relational import (Eq, Ge, Gt, Le, Lt, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.integers import (ceiling, floor, frac)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin, cos, tan
from sympy.core.expr import unchanged
from sympy.testing.pytest import XFAIL
def test_frac_leading_term():
    assert frac(x).as_leading_term(x) == x
    assert frac(x).as_leading_term(x, cdir=1) == x
    assert frac(x).as_leading_term(x, cdir=-1) == 1
    assert frac(x + S.Half).as_leading_term(x, cdir=1) == S.Half
    assert frac(x + S.Half).as_leading_term(x, cdir=-1) == S.Half
    assert frac(-2 * x + 1).as_leading_term(x, cdir=1) == S.One
    assert frac(-2 * x + 1).as_leading_term(x, cdir=-1) == -2 * x
    assert frac(sin(x) + 5).as_leading_term(x, cdir=1) == x
    assert frac(sin(x) + 5).as_leading_term(x, cdir=-1) == S.One
    assert frac(sin(x ** 2) + 5).as_leading_term(x, cdir=1) == x ** 2
    assert frac(sin(x ** 2) + 5).as_leading_term(x, cdir=-1) == x ** 2