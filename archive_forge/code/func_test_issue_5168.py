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
def test_issue_5168():
    a, b, c = symbols('a b c', cls=Wild)
    x = Symbol('x')
    f = Function('f')
    assert x.match(a) == {a: x}
    assert x.match(a * f(x) ** c) == {a: x, c: 0}
    assert x.match(a * b) == {a: 1, b: x}
    assert x.match(a * b * f(x) ** c) == {a: 1, b: x, c: 0}
    assert (-x).match(a) == {a: -x}
    assert (-x).match(a * f(x) ** c) == {a: -x, c: 0}
    assert (-x).match(a * b) == {a: -1, b: x}
    assert (-x).match(a * b * f(x) ** c) == {a: -1, b: x, c: 0}
    assert (2 * x).match(a) == {a: 2 * x}
    assert (2 * x).match(a * f(x) ** c) == {a: 2 * x, c: 0}
    assert (2 * x).match(a * b) == {a: 2, b: x}
    assert (2 * x).match(a * b * f(x) ** c) == {a: 2, b: x, c: 0}
    assert (-2 * x).match(a) == {a: -2 * x}
    assert (-2 * x).match(a * f(x) ** c) == {a: -2 * x, c: 0}
    assert (-2 * x).match(a * b) == {a: -2, b: x}
    assert (-2 * x).match(a * b * f(x) ** c) == {a: -2, b: x, c: 0}