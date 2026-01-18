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
def test_meijerg_shift_operators():
    a1, a2, a3, a4, a5, b1, b2, b3, b4, b5 = (randcplx(n) for n in range(10))
    g = meijerg([a1], [a3, a4], [b1], [b3, b4], z)
    assert tn(MeijerShiftA(b1).apply(g, op), meijerg([a1], [a3, a4], [b1 + 1], [b3, b4], z), z)
    assert tn(MeijerShiftB(a1).apply(g, op), meijerg([a1 - 1], [a3, a4], [b1], [b3, b4], z), z)
    assert tn(MeijerShiftC(b3).apply(g, op), meijerg([a1], [a3, a4], [b1], [b3 + 1, b4], z), z)
    assert tn(MeijerShiftD(a3).apply(g, op), meijerg([a1], [a3 - 1, a4], [b1], [b3, b4], z), z)
    s = MeijerUnShiftA([a1], [a3, a4], [b1], [b3, b4], 0, z)
    assert tn(s.apply(g, op), meijerg([a1], [a3, a4], [b1 - 1], [b3, b4], z), z)
    s = MeijerUnShiftC([a1], [a3, a4], [b1], [b3, b4], 0, z)
    assert tn(s.apply(g, op), meijerg([a1], [a3, a4], [b1], [b3 - 1, b4], z), z)
    s = MeijerUnShiftB([a1], [a3, a4], [b1], [b3, b4], 0, z)
    assert tn(s.apply(g, op), meijerg([a1 + 1], [a3, a4], [b1], [b3, b4], z), z)
    s = MeijerUnShiftD([a1], [a3, a4], [b1], [b3, b4], 0, z)
    assert tn(s.apply(g, op), meijerg([a1], [a3 + 1, a4], [b1], [b3, b4], z), z)