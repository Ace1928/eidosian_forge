from sympy.polys.partfrac import (
from sympy.core.expr import Expr
from sympy.core.function import Lambda
from sympy.core.numbers import (E, I, Rational, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import Matrix
from sympy.polys.polytools import (Poly, factor)
from sympy.polys.rationaltools import together
from sympy.polys.rootoftools import RootSum
from sympy.testing.pytest import raises, XFAIL
from sympy.abc import x, y, a, b, c
def test_apart_symbolic():
    f = a * x ** 4 + (2 * b + 2 * a * c) * x ** 3 + (4 * b * c - a ** 2 + a * c ** 2) * x ** 2 + (-2 * a * b + 2 * b * c ** 2) * x - b ** 2
    g = a ** 2 * x ** 4 + (2 * a * b + 2 * c * a ** 2) * x ** 3 + (4 * a * b * c + b ** 2 + a ** 2 * c ** 2) * x ** 2 + (2 * c * b ** 2 + 2 * a * b * c ** 2) * x + b ** 2 * c ** 2
    assert apart(f / g, x) == 1 / a - 1 / (x + c) ** 2 - b ** 2 / (a * (a * x + b) ** 2)
    assert apart(1 / ((x + a) * (x + b) * (x + c)), x) == 1 / ((a - c) * (b - c) * (c + x)) - 1 / ((a - b) * (b - c) * (b + x)) + 1 / ((a - b) * (a - c) * (a + x))