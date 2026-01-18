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
def test_mul_noncommutative_pow():
    A, B, C = symbols('A B C', commutative=False)
    w = symbols('w', cls=Wild, commutative=False)
    assert (A * B * w).matches(A * B ** 2) == {w: B}
    assert (A * B ** 2 * w * B ** 3).matches(A * B ** 8) == {w: B ** 3}
    assert (A * B * w * C).matches(A * B ** 4 * C) == {w: B ** 3}
    assert (A * B * w ** (-1)).matches(A * B * C ** (-1)) == {w: C}
    assert (A * (B * w) ** (-1) * C).matches(A * (B * C) ** (-1) * C) == {w: C}
    assert (w ** 2 * B * C).matches(A ** 2 * B * C) == {w: A}
    assert (w ** 2 * B * w ** 3).matches(A ** 2 * B * A ** 3) == {w: A}
    assert (w ** 2 * B * w ** 4).matches(A ** 2 * B * A ** 2) is None