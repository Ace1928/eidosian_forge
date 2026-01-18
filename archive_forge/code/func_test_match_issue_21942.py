from sympy import abc
from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.function import (Derivative, Function, diff)
from sympy.core.mul import Mul
from sympy.core.numbers import (Float, I, Integer, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, Wild, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.hyper import meijerg
from sympy.polys.polytools import Poly
from sympy.simplify.radsimp import collect
from sympy.simplify.simplify import signsimp
from sympy.testing.pytest import XFAIL
def test_match_issue_21942():
    a, r, w = symbols('a, r, w', nonnegative=True)
    p = symbols('p', positive=True)
    g_ = Wild('g')
    pattern = g_ ** (1 / (1 - p))
    eq = (a * r ** (1 - p) + w ** (1 - p) * (1 - a)) ** (1 / (1 - p))
    m = {g_: a * r ** (1 - p) + w ** (1 - p) * (1 - a)}
    assert pattern.matches(eq) == m
    assert (-pattern).matches(-eq) == m
    assert pattern.matches(signsimp(eq)) is None