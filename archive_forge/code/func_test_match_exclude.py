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
def test_match_exclude():
    x = Symbol('x')
    y = Symbol('y')
    p = Wild('p')
    q = Wild('q')
    r = Wild('r')
    e = Rational(6)
    assert e.match(2 * p) == {p: 3}
    e = 3 / (4 * x + 5)
    assert e.match(3 / (p * x + q)) == {p: 4, q: 5}
    e = 3 / (4 * x + 5)
    assert e.match(p / (q * x + r)) == {p: 3, q: 4, r: 5}
    e = 2 / (x + 1)
    assert e.match(p / (q * x + r)) == {p: 2, q: 1, r: 1}
    e = 1 / (x + 1)
    assert e.match(p / (q * x + r)) == {p: 1, q: 1, r: 1}
    e = 4 * x + 5
    assert e.match(p * x + q) == {p: 4, q: 5}
    e = 4 * x + 5 * y + 6
    assert e.match(p * x + q * y + r) == {p: 4, q: 5, r: 6}
    a = Wild('a', exclude=[x])
    e = 3 * x
    assert e.match(p * x) == {p: 3}
    assert e.match(a * x) == {a: 3}
    e = 3 * x ** 2
    assert e.match(p * x) == {p: 3 * x}
    assert e.match(a * x) is None
    e = 3 * x + 3 + 6 / x
    assert e.match(p * x ** 2 + p * x + 2 * p) == {p: 3 / x}
    assert e.match(a * x ** 2 + a * x + 2 * a) is None