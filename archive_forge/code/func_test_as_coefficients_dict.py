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
def test_as_coefficients_dict():
    check = [S.One, x, y, x * y, 1]
    assert [Add(3 * x, 2 * x, y, 3).as_coefficients_dict()[i] for i in check] == [3, 5, 1, 0, 3]
    assert [Add(3 * x, 2 * x, y, 3, evaluate=False).as_coefficients_dict()[i] for i in check] == [3, 5, 1, 0, 3]
    assert [(3 * x * y).as_coefficients_dict()[i] for i in check] == [0, 0, 0, 3, 0]
    assert [(3.0 * x * y).as_coefficients_dict()[i] for i in check] == [0, 0, 0, 3.0, 0]
    assert (3.0 * x * y).as_coefficients_dict()[3.0 * x * y] == 0
    eq = x * (x + 1) * a + x * b + c / x
    assert eq.as_coefficients_dict(x) == {x: b, 1 / x: c, x * (x + 1): a}
    assert eq.expand().as_coefficients_dict(x) == {x ** 2: a, x: a + b, 1 / x: c}
    assert x.as_coefficients_dict() == {x: S.One}