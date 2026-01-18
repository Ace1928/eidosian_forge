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
def test_as_independent():
    assert S.Zero.as_independent(x, as_Add=True) == (0, 0)
    assert S.Zero.as_independent(x, as_Add=False) == (0, 0)
    assert (2 * x * sin(x) + y + x).as_independent(x) == (y, x + 2 * x * sin(x))
    assert (2 * x * sin(x) + y + x).as_independent(y) == (x + 2 * x * sin(x), y)
    assert (2 * x * sin(x) + y + x).as_independent(x, y) == (0, y + x + 2 * x * sin(x))
    assert (x * sin(x) * cos(y)).as_independent(x) == (cos(y), x * sin(x))
    assert (x * sin(x) * cos(y)).as_independent(y) == (x * sin(x), cos(y))
    assert (x * sin(x) * cos(y)).as_independent(x, y) == (1, x * sin(x) * cos(y))
    assert sin(x).as_independent(x) == (1, sin(x))
    assert sin(x).as_independent(y) == (sin(x), 1)
    assert (2 * sin(x)).as_independent(x) == (2, sin(x))
    assert (2 * sin(x)).as_independent(y) == (2 * sin(x), 1)
    n1, n2, n3 = symbols('n1 n2 n3', commutative=False)
    assert (n1 + n1 * n2).as_independent(n2) == (n1, n1 * n2)
    assert (n2 * n1 + n1 * n2).as_independent(n2) == (0, n1 * n2 + n2 * n1)
    assert (n1 * n2 * n1).as_independent(n2) == (n1, n2 * n1)
    assert (n1 * n2 * n1).as_independent(n1) == (1, n1 * n2 * n1)
    assert (3 * x).as_independent(x, as_Add=True) == (0, 3 * x)
    assert (3 * x).as_independent(x, as_Add=False) == (3, x)
    assert (3 + x).as_independent(x, as_Add=True) == (3, x)
    assert (3 + x).as_independent(x, as_Add=False) == (1, 3 + x)
    assert (3 * x).as_independent(Symbol) == (3, x)
    assert (n1 * x * y).as_independent(x) == (n1 * y, x)
    assert ((x + n1) * (x - y)).as_independent(x) == (1, (x + n1) * (x - y))
    assert ((x + n1) * (x - y)).as_independent(y) == (x + n1, x - y)
    assert (DiracDelta(x - n1) * DiracDelta(x - y)).as_independent(x) == (1, DiracDelta(x - n1) * DiracDelta(x - y))
    assert (x * y * n1 * n2 * n3).as_independent(n2) == (x * y * n1, n2 * n3)
    assert (x * y * n1 * n2 * n3).as_independent(n1) == (x * y, n1 * n2 * n3)
    assert (x * y * n1 * n2 * n3).as_independent(n3) == (x * y * n1 * n2, n3)
    assert (DiracDelta(x - n1) * DiracDelta(y - n1) * DiracDelta(x - n2)).as_independent(y) == (DiracDelta(x - n1) * DiracDelta(x - n2), DiracDelta(y - n1))
    assert (x + Integral(x, (x, 1, 2))).as_independent(x, strict=True) == (Integral(x, (x, 1, 2)), x)
    eq = Add(x, -x, 2, -3, evaluate=False)
    assert eq.as_independent(x) == (-1, Add(x, -x, evaluate=False))
    eq = Mul(x, 1 / x, 2, -3, evaluate=False)
    assert eq.as_independent(x) == (-6, Mul(x, 1 / x, evaluate=False))
    assert (x * y).as_independent(z, as_Add=True) == (x * y, 0)