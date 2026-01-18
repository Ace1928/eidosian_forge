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
def test_HolonomicFunction_multiplication():
    x = symbols('x')
    R, Dx = DifferentialOperators(ZZ.old_poly_ring(x), 'Dx')
    p = HolonomicFunction(Dx + x + x * Dx ** 2, x)
    q = HolonomicFunction(x * Dx + Dx * x + Dx ** 2, x)
    r = HolonomicFunction(8 * x ** 6 + 4 * x ** 4 + 6 * x ** 2 + 3 + (24 * x ** 5 - 4 * x ** 3 + 24 * x) * Dx + (8 * x ** 6 + 20 * x ** 4 + 12 * x ** 2 + 2) * Dx ** 2 + (8 * x ** 5 + 4 * x ** 3 + 4 * x) * Dx ** 3 + (2 * x ** 4 + x ** 2) * Dx ** 4, x)
    assert p * q == r
    p = HolonomicFunction(Dx ** 2 + 1, x)
    q = HolonomicFunction(Dx - 1, x)
    r = HolonomicFunction(2 + -2 * Dx + 1 * Dx ** 2, x)
    assert p * q == r
    p = HolonomicFunction(Dx ** 2 + 1 + x + Dx, x)
    q = HolonomicFunction((Dx * x - 1) ** 2, x)
    r = HolonomicFunction(4 * x ** 7 + 11 * x ** 6 + 16 * x ** 5 + 4 * x ** 4 - 6 * x ** 3 - 7 * x ** 2 - 8 * x - 2 + (8 * x ** 6 + 26 * x ** 5 + 24 * x ** 4 - 3 * x ** 3 - 11 * x ** 2 - 6 * x - 2) * Dx + (8 * x ** 6 + 18 * x ** 5 + 15 * x ** 4 - 3 * x ** 3 - 6 * x ** 2 - 6 * x - 2) * Dx ** 2 + (8 * x ** 5 + 10 * x ** 4 + 6 * x ** 3 - 2 * x ** 2 - 4 * x) * Dx ** 3 + (4 * x ** 5 + 3 * x ** 4 - x ** 2) * Dx ** 4, x)
    assert p * q == r
    p = HolonomicFunction(x * Dx ** 2 - 1, x)
    q = HolonomicFunction(Dx * x - x, x)
    r = HolonomicFunction(x - 3 + (-2 * x + 2) * Dx + x * Dx ** 2, x)
    assert p * q == r