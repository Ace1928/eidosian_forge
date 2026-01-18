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
def test_from_meijerg():
    x = symbols('x')
    R, Dx = DifferentialOperators(QQ.old_poly_ring(x), 'Dx')
    p = from_meijerg(meijerg(([], [Rational(3, 2)]), ([S.Half], [S.Half, 1]), x))
    q = HolonomicFunction(x / 2 - Rational(1, 4) + (-x ** 2 + x / 4) * Dx + x ** 2 * Dx ** 2 + x ** 3 * Dx ** 3, x, 1, [1 / sqrt(pi), 1 / (2 * sqrt(pi)), -1 / (4 * sqrt(pi))])
    assert p == q
    p = from_meijerg(meijerg(([], []), ([0], []), x))
    q = HolonomicFunction(1 + Dx, x, 0, [1])
    assert p == q
    p = from_meijerg(meijerg(([1], []), ([S.Half], [0]), x))
    q = HolonomicFunction((x + S.Half) * Dx + x * Dx ** 2, x, 1, [sqrt(pi) * erf(1), exp(-1)])
    assert p == q
    p = from_meijerg(meijerg(([0], [1]), ([0], []), 2 * x ** 2))
    q = HolonomicFunction((3 * x ** 2 - 1) * Dx + x ** 3 * Dx ** 2, x, 1, [-exp(Rational(-1, 2)) + 1, -exp(Rational(-1, 2))])
    assert p == q