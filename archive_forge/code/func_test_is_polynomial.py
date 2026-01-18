from sympy.assumptions.refine import refine
from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.expr import (ExprBuilder, unchanged, Expr,
from sympy.core.function import (Function, expand, WildFunction,
from sympy.core.mul import Mul
from sympy.core.numbers import (NumberSymbol, E, zoo, oo, Float, I,
from sympy.core.power import Pow
from sympy.core.relational import Ge, Lt, Gt, Le
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Symbol, symbols, Dummy, Wild
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import exp_polar, exp, log
from sympy.functions.elementary.miscellaneous import sqrt, Max
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import tan, sin, cos
from sympy.functions.special.delta_functions import (Heaviside,
from sympy.functions.special.error_functions import Si
from sympy.functions.special.gamma_functions import gamma
from sympy.integrals.integrals import integrate, Integral
from sympy.physics.secondquant import FockState
from sympy.polys.partfrac import apart
from sympy.polys.polytools import factor, cancel, Poly
from sympy.polys.rationaltools import together
from sympy.series.order import O
from sympy.sets.sets import FiniteSet
from sympy.simplify.combsimp import combsimp
from sympy.simplify.gammasimp import gammasimp
from sympy.simplify.powsimp import powsimp
from sympy.simplify.radsimp import collect, radsimp
from sympy.simplify.ratsimp import ratsimp
from sympy.simplify.simplify import simplify, nsimplify
from sympy.simplify.trigsimp import trigsimp
from sympy.tensor.indexed import Indexed
from sympy.physics.units import meter
from sympy.testing.pytest import raises, XFAIL
from sympy.abc import a, b, c, n, t, u, x, y, z
def test_is_polynomial():
    k = Symbol('k', nonnegative=True, integer=True)
    assert Rational(2).is_polynomial(x, y, z) is True
    assert S.Pi.is_polynomial(x, y, z) is True
    assert x.is_polynomial(x) is True
    assert x.is_polynomial(y) is True
    assert (x ** 2).is_polynomial(x) is True
    assert (x ** 2).is_polynomial(y) is True
    assert (x ** (-2)).is_polynomial(x) is False
    assert (x ** (-2)).is_polynomial(y) is True
    assert (2 ** x).is_polynomial(x) is False
    assert (2 ** x).is_polynomial(y) is True
    assert (x ** k).is_polynomial(x) is False
    assert (x ** k).is_polynomial(k) is False
    assert (x ** x).is_polynomial(x) is False
    assert (k ** k).is_polynomial(k) is False
    assert (k ** x).is_polynomial(k) is False
    assert (x ** (-k)).is_polynomial(x) is False
    assert ((2 * x) ** k).is_polynomial(x) is False
    assert (x ** 2 + 3 * x - 8).is_polynomial(x) is True
    assert (x ** 2 + 3 * x - 8).is_polynomial(y) is True
    assert (x ** 2 + 3 * x - 8).is_polynomial() is True
    assert sqrt(x).is_polynomial(x) is False
    assert (sqrt(x) ** 3).is_polynomial(x) is False
    assert (x ** 2 + 3 * x * sqrt(y) - 8).is_polynomial(x) is True
    assert (x ** 2 + 3 * x * sqrt(y) - 8).is_polynomial(y) is False
    assert (x ** 2 * y ** 2 + x * y ** 2 + y * x + exp(2)).is_polynomial() is True
    assert (x ** 2 * y ** 2 + x * y ** 2 + y * x + exp(x)).is_polynomial() is False
    assert (x ** 2 * y ** 2 + x * y ** 2 + y * x + exp(2)).is_polynomial(x, y) is True
    assert (x ** 2 * y ** 2 + x * y ** 2 + y * x + exp(x)).is_polynomial(x, y) is False
    assert (1 / f(x) + 1).is_polynomial(f(x)) is False