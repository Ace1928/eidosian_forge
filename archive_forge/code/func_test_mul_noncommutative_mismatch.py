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
def test_mul_noncommutative_mismatch():
    A, B, C = symbols('A B C', commutative=False)
    w = symbols('w', cls=Wild, commutative=False)
    assert (w * B * w).matches(A * B * A) == {w: A}
    assert (w * B * w).matches(A * C * B * A * C) == {w: A * C}
    assert (w * B * w).matches(A * C * B * A * B) is None
    assert (w * B * w).matches(A * B * C) is None
    assert (w * w * C).matches(A * B * C) is None