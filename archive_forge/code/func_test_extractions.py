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
def test_extractions():
    for base in (2, S.Exp1):
        assert Pow(base ** x, 3, evaluate=False).extract_multiplicatively(base ** x) == base ** (2 * x)
        assert (base ** (5 * x)).extract_multiplicatively(base ** (3 * x)) == base ** (2 * x)
    assert ((x * y) ** 3).extract_multiplicatively(x ** 2 * y) == x * y ** 2
    assert ((x * y) ** 3).extract_multiplicatively(x ** 4 * y) is None
    assert (2 * x).extract_multiplicatively(2) == x
    assert (2 * x).extract_multiplicatively(3) is None
    assert (2 * x).extract_multiplicatively(-1) is None
    assert (S.Half * x).extract_multiplicatively(3) == x / 6
    assert sqrt(x).extract_multiplicatively(x) is None
    assert sqrt(x).extract_multiplicatively(1 / x) is None
    assert x.extract_multiplicatively(-x) is None
    assert (-2 - 4 * I).extract_multiplicatively(-2) == 1 + 2 * I
    assert (-2 - 4 * I).extract_multiplicatively(3) is None
    assert (-2 * x - 4 * y - 8).extract_multiplicatively(-2) == x + 2 * y + 4
    assert (-2 * x * y - 4 * x ** 2 * y).extract_multiplicatively(-2 * y) == 2 * x ** 2 + x
    assert (2 * x * y + 4 * x ** 2 * y).extract_multiplicatively(2 * y) == 2 * x ** 2 + x
    assert (-4 * y ** 2 * x).extract_multiplicatively(-3 * y) is None
    assert (2 * x).extract_multiplicatively(1) == 2 * x
    assert (-oo).extract_multiplicatively(5) is -oo
    assert oo.extract_multiplicatively(5) is oo
    assert ((x * y) ** 3).extract_additively(1) is None
    assert (x + 1).extract_additively(x) == 1
    assert (x + 1).extract_additively(2 * x) is None
    assert (x + 1).extract_additively(-x) is None
    assert (-x + 1).extract_additively(2 * x) is None
    assert (2 * x + 3).extract_additively(x) == x + 3
    assert (2 * x + 3).extract_additively(2) == 2 * x + 1
    assert (2 * x + 3).extract_additively(3) == 2 * x
    assert (2 * x + 3).extract_additively(-2) is None
    assert (2 * x + 3).extract_additively(3 * x) is None
    assert (2 * x + 3).extract_additively(2 * x) == 3
    assert x.extract_additively(0) == x
    assert S(2).extract_additively(x) is None
    assert S(2.0).extract_additively(2.0) is S.Zero
    assert S(2.0).extract_additively(2) is S.Zero
    assert S(2 * x + 3).extract_additively(x + 1) == x + 2
    assert S(2 * x + 3).extract_additively(y + 1) is None
    assert S(2 * x - 3).extract_additively(x + 1) is None
    assert S(2 * x - 3).extract_additively(y + z) is None
    assert ((a + 1) * x * 4 + y).extract_additively(x).expand() == 4 * a * x + 3 * x + y
    assert ((a + 1) * x * 4 + 3 * y).extract_additively(x + 2 * y).expand() == 4 * a * x + 3 * x + y
    assert (y * (x + 1)).extract_additively(x + 1) is None
    assert ((y + 1) * (x + 1) + 3).extract_additively(x + 1) == y * (x + 1) + 3
    assert ((x + y) * (x + 1) + x + y + 3).extract_additively(x + y) == x * (x + y) + 3
    assert (x + y + 2 * ((x + y) * (x + 1)) + 3).extract_additively((x + y) * (x + 1)) == x + y + (x + 1) * (x + y) + 3
    assert ((y + 1) * (x + 2 * y + 1) + 3).extract_additively(y + 1) == (x + 2 * y) * (y + 1) + 3
    assert (-x - x * I).extract_additively(-x) == -I * x
    assert (4 * x * (y + 1) + y).extract_additively(x) == x * (4 * y + 3) + y
    n = Symbol('n', integer=True)
    assert Integer(-3).could_extract_minus_sign() is True
    assert (-n * x + x).could_extract_minus_sign() != (n * x - x).could_extract_minus_sign()
    assert (x - y).could_extract_minus_sign() != (-x + y).could_extract_minus_sign()
    assert (1 - x - y).could_extract_minus_sign() is True
    assert (1 - x + y).could_extract_minus_sign() is False
    assert ((-x - x * y) / y).could_extract_minus_sign() is False
    assert ((x + x * y) / -y).could_extract_minus_sign() is True
    assert ((x + x * y) / y).could_extract_minus_sign() is False
    assert ((-x - y) / (x + y)).could_extract_minus_sign() is False

    class sign_invariant(Function, Expr):
        nargs = 1

        def __neg__(self):
            return self
    foo = sign_invariant(x)
    assert foo == -foo
    assert foo.could_extract_minus_sign() is False
    assert (x - y).could_extract_minus_sign() is False
    assert (-x + y).could_extract_minus_sign() is True
    assert (x - 1).could_extract_minus_sign() is False
    assert (1 - x).could_extract_minus_sign() is True
    assert (sqrt(2) - 1).could_extract_minus_sign() is True
    assert (1 - sqrt(2)).could_extract_minus_sign() is False
    eq = (3 * x + 15 * y).extract_multiplicatively(3)
    assert eq.args == eq.func(*eq.args).args