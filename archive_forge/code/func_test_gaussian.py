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
def test_gaussian():
    mu, x = symbols('mu x')
    sd = symbols('sd', positive=True)
    Q = QQ[mu, sd].get_field()
    e = sqrt(2) * exp(-(-mu + x) ** 2 / (2 * sd ** 2)) / (2 * sqrt(pi) * sd)
    h1 = expr_to_holonomic(e, x, domain=Q)
    _, Dx = DifferentialOperators(Q.old_poly_ring(x), 'Dx')
    h2 = HolonomicFunction(-mu / sd ** 2 + x / sd ** 2 + 1 * Dx, x)
    assert h1 == h2