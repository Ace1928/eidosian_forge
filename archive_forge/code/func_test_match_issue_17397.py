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
def test_match_issue_17397():
    f = Function('f')
    x = Symbol('x')
    a3 = Wild('a3', exclude=[f(x), f(x).diff(x), f(x).diff(x, 2)])
    b3 = Wild('b3', exclude=[f(x), f(x).diff(x), f(x).diff(x, 2)])
    c3 = Wild('c3', exclude=[f(x), f(x).diff(x), f(x).diff(x, 2)])
    deq = a3 * f(x).diff(x, 2) + b3 * f(x).diff(x) + c3 * f(x)
    eq = (x - 2) ** 2 * f(x).diff(x, 2) + (x - 2) * f(x).diff(x) + ((x - 2) ** 2 - 4) * f(x)
    r = collect(eq, [f(x).diff(x, 2), f(x).diff(x), f(x)]).match(deq)
    assert r == {a3: (x - 2) ** 2, c3: (x - 2) ** 2 - 4, b3: x - 2}
    eq = x * f(x) + x * Derivative(f(x), (x, 2)) - 4 * f(x) + Derivative(f(x), x) - 4 * Derivative(f(x), (x, 2)) - 2 * Derivative(f(x), x) / x + 4 * Derivative(f(x), (x, 2)) / x
    r = collect(eq, [f(x).diff(x, 2), f(x).diff(x), f(x)]).match(deq)
    assert r == {a3: x - 4 + 4 / x, b3: 1 - 2 / x, c3: x - 4}