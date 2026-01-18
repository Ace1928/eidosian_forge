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
def test_as_coeff_exponent():
    assert (3 * x ** 4).as_coeff_exponent(x) == (3, 4)
    assert (2 * x ** 3).as_coeff_exponent(x) == (2, 3)
    assert (4 * x ** 2).as_coeff_exponent(x) == (4, 2)
    assert (6 * x ** 1).as_coeff_exponent(x) == (6, 1)
    assert (3 * x ** 0).as_coeff_exponent(x) == (3, 0)
    assert (2 * x ** 0).as_coeff_exponent(x) == (2, 0)
    assert (1 * x ** 0).as_coeff_exponent(x) == (1, 0)
    assert (0 * x ** 0).as_coeff_exponent(x) == (0, 0)
    assert (-1 * x ** 0).as_coeff_exponent(x) == (-1, 0)
    assert (-2 * x ** 0).as_coeff_exponent(x) == (-2, 0)
    assert (2 * x ** 3 + pi * x ** 3).as_coeff_exponent(x) == (2 + pi, 3)
    assert (x * log(2) / (2 * x + pi * x)).as_coeff_exponent(x) == (log(2) / (2 + pi), 0)
    D = Derivative
    fx = D(f(x), x)
    assert fx.as_coeff_exponent(f(x)) == (fx, 0)