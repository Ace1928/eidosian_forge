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
def test_match_deriv_bug1():
    n = Function('n')
    l = Function('l')
    x = Symbol('x')
    p = Wild('p')
    e = diff(l(x), x) / x - diff(diff(n(x), x), x) / 2 - diff(n(x), x) ** 2 / 4 + diff(n(x), x) * diff(l(x), x) / 4
    e = e.subs(n(x), -l(x)).doit()
    t = x * exp(-l(x))
    t2 = t.diff(x, x) / t
    assert e.match((p * t2).expand()) == {p: Rational(-1, 2)}