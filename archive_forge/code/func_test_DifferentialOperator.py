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
def test_DifferentialOperator():
    x = symbols('x')
    R, Dx = DifferentialOperators(QQ.old_poly_ring(x), 'Dx')
    assert Dx == R.derivative_operator
    assert Dx == DifferentialOperator([R.base.zero, R.base.one], R)
    assert x * Dx + x ** 2 * Dx ** 2 == DifferentialOperator([0, x, x ** 2], R)
    assert x ** 2 + 1 + Dx + x * Dx ** 5 == DifferentialOperator([x ** 2 + 1, 1, 0, 0, 0, x], R)
    assert (x * Dx + x ** 2 + 1 - Dx * (x ** 3 + x)) ** 3 == -48 * x ** 6 + -57 * x ** 7 * Dx + -15 * x ** 8 * Dx ** 2 + -x ** 9 * Dx ** 3
    p = (x * Dx ** 2 + (x ** 2 + 3) * Dx ** 5) * (Dx + x ** 2)
    q = 2 * x + 4 * x ** 2 * Dx + x ** 3 * Dx ** 2 + (20 * x ** 2 + x + 60) * Dx ** 3 + (10 * x ** 3 + 30 * x) * Dx ** 4 + (x ** 4 + 3 * x ** 2) * Dx ** 5 + (x ** 2 + 3) * Dx ** 6
    assert p == q