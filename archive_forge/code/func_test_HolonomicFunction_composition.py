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
def test_HolonomicFunction_composition():
    x = symbols('x')
    R, Dx = DifferentialOperators(ZZ.old_poly_ring(x), 'Dx')
    p = HolonomicFunction(Dx - 1, x).composition(x ** 2 + x)
    r = HolonomicFunction(-2 * x - 1 + Dx, x)
    assert p == r
    p = HolonomicFunction(Dx ** 2 + 1, x).composition(x ** 5 + x ** 2 + 1)
    r = HolonomicFunction(125 * x ** 12 + 150 * x ** 9 + 60 * x ** 6 + 8 * x ** 3 + (-20 * x ** 3 - 2) * Dx + (5 * x ** 4 + 2 * x) * Dx ** 2, x)
    assert p == r
    p = HolonomicFunction(Dx ** 2 * x + x, x).composition(2 * x ** 3 + x ** 2 + 1)
    r = HolonomicFunction(216 * x ** 9 + 324 * x ** 8 + 180 * x ** 7 + 152 * x ** 6 + 112 * x ** 5 + 36 * x ** 4 + 4 * x ** 3 + (24 * x ** 4 + 16 * x ** 3 + 3 * x ** 2 - 6 * x - 1) * Dx + (6 * x ** 5 + 5 * x ** 4 + x ** 3 + 3 * x ** 2 + x) * Dx ** 2, x)
    assert p == r
    p = HolonomicFunction(Dx ** 2 + 1, x).composition(1 - x ** 2)
    r = HolonomicFunction(4 * x ** 3 - Dx + x * Dx ** 2, x)
    assert p == r
    p = HolonomicFunction(Dx ** 2 + 1, x).composition(x - 2 / (x ** 2 + 1))
    r = HolonomicFunction(x ** 12 + 6 * x ** 10 + 12 * x ** 9 + 15 * x ** 8 + 48 * x ** 7 + 68 * x ** 6 + 72 * x ** 5 + 111 * x ** 4 + 112 * x ** 3 + 54 * x ** 2 + 12 * x + 1 + (12 * x ** 8 + 32 * x ** 6 + 24 * x ** 4 - 4) * Dx + (x ** 12 + 6 * x ** 10 + 4 * x ** 9 + 15 * x ** 8 + 16 * x ** 7 + 20 * x ** 6 + 24 * x ** 5 + 15 * x ** 4 + 16 * x ** 3 + 6 * x ** 2 + 4 * x + 1) * Dx ** 2, x)
    assert p == r