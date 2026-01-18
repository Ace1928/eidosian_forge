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
def test_as_leading_term():
    assert (3 + 2 * x ** (log(3) / log(2) - 1)).as_leading_term(x) == 3
    assert (1 / x ** 2 + 1 + x + x ** 2).as_leading_term(x) == 1 / x ** 2
    assert (1 / x + 1 + x + x ** 2).as_leading_term(x) == 1 / x
    assert (x ** 2 + 1 / x).as_leading_term(x) == 1 / x
    assert (1 + x ** 2).as_leading_term(x) == 1
    assert (x + 1).as_leading_term(x) == 1
    assert (x + x ** 2).as_leading_term(x) == x
    assert (x ** 2).as_leading_term(x) == x ** 2
    assert (x + oo).as_leading_term(x) is oo
    raises(ValueError, lambda: (x + 1).as_leading_term(1))
    e = -3 * x + (x + Rational(3, 2) - sqrt(3) * S.ImaginaryUnit / 2) ** 2 - Rational(3, 2) + 3 * sqrt(3) * S.ImaginaryUnit / 2
    assert e.as_leading_term(x) == (12 * sqrt(3) * x - 12 * S.ImaginaryUnit * x) / (4 * sqrt(3) + 12 * S.ImaginaryUnit)
    e = 1 - x - x ** 2
    d = (1 + sqrt(5)) / 2
    assert e.subs(x, y + 1 / d).as_leading_term(y) == (-576 * sqrt(5) * y - 1280 * y) / (256 * sqrt(5) + 576)