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
def test_match_terms():
    X, Y = map(Wild, 'XY')
    x, y, z = symbols('x y z')
    assert (5 * y - x).match(5 * X - Y) == {X: y, Y: x}
    assert (x + (y - 1) * z).match(x + X * z) == {X: y - 1}
    assert (x - log(x / y) * (1 - exp(x / y))).match(x - log(X / y) * (1 - exp(x / y))) == {X: x}