from sympy.holonomic import (DifferentialOperator, HolonomicFunction,
from sympy.holonomic.recurrence import RecurrenceOperators, HolonomicSequence
from sympy.core import EulerGamma
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (asinh, cosh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.bessel import besselj
from sympy.functions.special.beta_functions import beta
from sympy.functions.special.error_functions import (Ci, Si, erf, erfc)
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.special.hyper import (hyper, meijerg)
from sympy.printing.str import sstr
from sympy.series.order import O
from sympy.simplify.hyperexpand import hyperexpand
from sympy.polys.domains.integerring import ZZ
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.domains.realfield import RR
def test_HolonomicFunction_addition():
    x = symbols('x')
    R, Dx = DifferentialOperators(ZZ.old_poly_ring(x), 'Dx')
    p = HolonomicFunction(Dx ** 2 * x, x)
    q = HolonomicFunction(2 * Dx + x * Dx ** 2, x)
    assert p == q
    p = HolonomicFunction(x * Dx + 1, x)
    q = HolonomicFunction(Dx + 1, x)
    r = HolonomicFunction(x - 2 + (x ** 2 - 2) * Dx + (x ** 2 - x) * Dx ** 2, x)
    assert p + q == r
    p = HolonomicFunction(x * Dx + Dx ** 2 * (x ** 2 + 2), x)
    q = HolonomicFunction(Dx - 3, x)
    r = HolonomicFunction(-54 * x ** 2 - 126 * x - 150 + (-135 * x ** 3 - 252 * x ** 2 - 270 * x + 140) * Dx + (-27 * x ** 4 - 24 * x ** 2 + 14 * x - 150) * Dx ** 2 + (9 * x ** 4 + 15 * x ** 3 + 38 * x ** 2 + 30 * x + 40) * Dx ** 3, x)
    assert p + q == r
    p = HolonomicFunction(Dx ** 5 - 1, x)
    q = HolonomicFunction(x ** 3 + Dx, x)
    r = HolonomicFunction(-x ** 18 + 45 * x ** 14 - 525 * x ** 10 + 1575 * x ** 6 - x ** 3 - 630 * x ** 2 + (-x ** 15 + 30 * x ** 11 - 195 * x ** 7 + 210 * x ** 3 - 1) * Dx + (x ** 18 - 45 * x ** 14 + 525 * x ** 10 - 1575 * x ** 6 + x ** 3 + 630 * x ** 2) * Dx ** 5 + (x ** 15 - 30 * x ** 11 + 195 * x ** 7 - 210 * x ** 3 + 1) * Dx ** 6, x)
    assert p + q == r
    p = x ** 2 + 3 * x + 8
    q = x ** 3 - 7 * x + 5
    p = p * Dx - p.diff()
    q = q * Dx - q.diff()
    r = HolonomicFunction(p, x) + HolonomicFunction(q, x)
    s = HolonomicFunction(6 * x ** 2 + 18 * x + 14 + (-4 * x ** 3 - 18 * x ** 2 - 62 * x + 10) * Dx + (x ** 4 + 6 * x ** 3 + 31 * x ** 2 - 10 * x - 71) * Dx ** 2, x)
    assert r == s