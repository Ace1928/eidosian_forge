from sympy.core.random import randrange
from sympy.simplify.hyperexpand import (ShiftA, ShiftB, UnShiftA, UnShiftB,
from sympy.concrete.summations import Sum
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.numbers import I
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.combinatorial.factorials import binomial
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.hyper import (hyper, meijerg)
from sympy.abc import z, a, b, c
from sympy.testing.pytest import XFAIL, raises, slow, ON_CI, skip
from sympy.core.random import verify_numerically as tn
from sympy.core.numbers import (Rational, pi)
from sympy.functions.elementary.exponential import (exp, exp_polar, log)
from sympy.functions.elementary.hyperbolic import atanh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (asin, cos, sin)
from sympy.functions.special.bessel import besseli
from sympy.functions.special.error_functions import erf
from sympy.functions.special.gamma_functions import (gamma, lowergamma)
@slow
def test_meijerg_confluence():

    def t(m, a, b):
        from sympy.core.sympify import sympify
        a, b = sympify([a, b])
        m_ = m
        m = hyperexpand(m)
        if not m == Piecewise((a, abs(z) < 1), (b, abs(1 / z) < 1), (m_, True)):
            return False
        if not (m.args[0].args[0] == a and m.args[1].args[0] == b):
            return False
        z0 = randcplx() / 10
        if abs(m.subs(z, z0).n() - a.subs(z, z0).n()).n() > 1e-10:
            return False
        if abs(m.subs(z, 1 / z0).n() - b.subs(z, 1 / z0).n()).n() > 1e-10:
            return False
        return True
    assert t(meijerg([], [1, 1], [0, 0], [], z), -log(z), 0)
    assert t(meijerg([], [3, 1], [0, 0], [], z), -z ** 2 / 4 + z - log(z) / 2 - Rational(3, 4), 0)
    assert t(meijerg([], [3, 1], [-1, 0], [], z), z ** 2 / 12 - z / 2 + log(z) / 2 + Rational(1, 4) + 1 / (6 * z), 0)
    assert t(meijerg([], [1, 1, 1, 1], [0, 0, 0, 0], [], z), -log(z) ** 3 / 6, 0)
    assert t(meijerg([1, 1], [], [], [0, 0], z), 0, -log(1 / z))
    assert t(meijerg([1, 1], [2, 2], [1, 1], [0, 0], z), -z * log(z) + 2 * z, -log(1 / z) + 2)
    assert t(meijerg([S.Half], [1, 1], [0, 0], [Rational(3, 2)], z), log(z) / 2 - 1, 0)

    def u(an, ap, bm, bq):
        m = meijerg(an, ap, bm, bq, z)
        m2 = hyperexpand(m, allow_hyper=True)
        if m2.has(meijerg) and (not (m2.is_Piecewise and len(m2.args) == 3)):
            return False
        return tn(m, m2, z)
    assert u([], [1], [0, 0], [])
    assert u([1, 1], [], [], [0])
    assert u([1, 1], [2, 2, 5], [1, 1, 6], [0, 0])
    assert u([1, 1], [2, 2, 5], [1, 1, 6], [0])