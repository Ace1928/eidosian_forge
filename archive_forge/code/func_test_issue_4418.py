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
def test_issue_4418():
    x = Symbol('x')
    a, b, c = symbols('a b c', cls=Wild, exclude=(x,))
    f, g = symbols('f g', cls=Function)
    eq = diff(g(x) * f(x).diff(x), x)
    assert eq.match(g(x).diff(x) * f(x).diff(x) + g(x) * f(x).diff(x, x) + c) == {c: 0}
    assert eq.match(a * g(x).diff(x) * f(x).diff(x) + b * g(x) * f(x).diff(x, x) + c) == {a: 1, b: 1, c: 0}