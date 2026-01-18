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
def test_as_ordered_terms():
    assert x.as_ordered_terms() == [x]
    assert (sin(x) ** 2 * cos(x) + sin(x) * cos(x) ** 2 + 1).as_ordered_terms() == [sin(x) ** 2 * cos(x), sin(x) * cos(x) ** 2, 1]
    args = [f(1), f(2), f(3), f(1, 2, 3), g(1), g(2), g(3), g(1, 2, 3)]
    expr = Add(*args)
    assert expr.as_ordered_terms() == args
    assert (1 + 4 * sqrt(3) * pi * x).as_ordered_terms() == [4 * pi * x * sqrt(3), 1]
    assert (2 + 3 * I).as_ordered_terms() == [2, 3 * I]
    assert (-2 + 3 * I).as_ordered_terms() == [-2, 3 * I]
    assert (2 - 3 * I).as_ordered_terms() == [2, -3 * I]
    assert (-2 - 3 * I).as_ordered_terms() == [-2, -3 * I]
    assert (4 + 3 * I).as_ordered_terms() == [4, 3 * I]
    assert (-4 + 3 * I).as_ordered_terms() == [-4, 3 * I]
    assert (4 - 3 * I).as_ordered_terms() == [4, -3 * I]
    assert (-4 - 3 * I).as_ordered_terms() == [-4, -3 * I]
    e = x ** 2 * y ** 2 + x * y ** 4 + y + 2
    assert e.as_ordered_terms(order='lex') == [x ** 2 * y ** 2, x * y ** 4, y, 2]
    assert e.as_ordered_terms(order='grlex') == [x * y ** 4, x ** 2 * y ** 2, y, 2]
    assert e.as_ordered_terms(order='rev-lex') == [2, y, x * y ** 4, x ** 2 * y ** 2]
    assert e.as_ordered_terms(order='rev-grlex') == [2, y, x ** 2 * y ** 2, x * y ** 4]
    k = symbols('k')
    assert k.as_ordered_terms(data=True) == ([(k, ((1.0, 0.0), (1,), ()))], [k])