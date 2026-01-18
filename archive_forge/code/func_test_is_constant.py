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
def test_is_constant():
    from sympy.solvers.solvers import checksol
    assert Sum(x, (x, 1, 10)).is_constant() is True
    assert Sum(x, (x, 1, n)).is_constant() is False
    assert Sum(x, (x, 1, n)).is_constant(y) is True
    assert Sum(x, (x, 1, n)).is_constant(n) is False
    assert Sum(x, (x, 1, n)).is_constant(x) is True
    eq = a * cos(x) ** 2 + a * sin(x) ** 2 - a
    assert eq.is_constant() is True
    assert eq.subs({x: pi, a: 2}) == eq.subs({x: pi, a: 3}) == 0
    assert x.is_constant() is False
    assert x.is_constant(y) is True
    assert log(x / y).is_constant() is False
    assert checksol(x, x, Sum(x, (x, 1, n))) is False
    assert checksol(x, x, Sum(x, (x, 1, n))) is False
    assert f(1).is_constant
    assert checksol(x, x, f(x)) is False
    assert Pow(x, S.Zero, evaluate=False).is_constant() is True
    assert Pow(S.Zero, x, evaluate=False).is_constant() is False
    assert (2 ** x).is_constant() is False
    assert Pow(S(2), S(3), evaluate=False).is_constant() is True
    z1, z2 = symbols('z1 z2', zero=True)
    assert (z1 + 2 * z2).is_constant() is True
    assert meter.is_constant() is True
    assert (3 * meter).is_constant() is True
    assert (x * meter).is_constant() is False