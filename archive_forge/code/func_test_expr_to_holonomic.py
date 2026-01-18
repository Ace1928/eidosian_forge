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
def test_expr_to_holonomic():
    x = symbols('x')
    R, Dx = DifferentialOperators(QQ.old_poly_ring(x), 'Dx')
    p = expr_to_holonomic((sin(x) / x) ** 2)
    q = HolonomicFunction(8 * x + (4 * x ** 2 + 6) * Dx + 6 * x * Dx ** 2 + x ** 2 * Dx ** 3, x, 0, [1, 0, Rational(-2, 3)])
    assert p == q
    p = expr_to_holonomic(1 / (1 + x ** 2) ** 2)
    q = HolonomicFunction(4 * x + (x ** 2 + 1) * Dx, x, 0, [1])
    assert p == q
    p = expr_to_holonomic(exp(x) * sin(x) + x * log(1 + x))
    q = HolonomicFunction(2 * x ** 3 + 10 * x ** 2 + 20 * x + 18 + (-2 * x ** 4 - 10 * x ** 3 - 20 * x ** 2 - 18 * x) * Dx + (2 * x ** 5 + 6 * x ** 4 + 7 * x ** 3 + 8 * x ** 2 + 10 * x - 4) * Dx ** 2 + (-2 * x ** 5 - 5 * x ** 4 - 2 * x ** 3 + 2 * x ** 2 - x + 4) * Dx ** 3 + (x ** 5 + 2 * x ** 4 - x ** 3 - 7 * x ** 2 / 2 + x + Rational(5, 2)) * Dx ** 4, x, 0, [0, 1, 4, -1])
    assert p == q
    p = expr_to_holonomic(x * exp(x) + cos(x) + 1)
    q = HolonomicFunction((-x - 3) * Dx + (x + 2) * Dx ** 2 + (-x - 3) * Dx ** 3 + (x + 2) * Dx ** 4, x, 0, [2, 1, 1, 3])
    assert p == q
    assert (x * exp(x) + cos(x) + 1).series(n=10) == p.series(n=10)
    p = expr_to_holonomic(log(1 + x) ** 2 + 1)
    q = HolonomicFunction(Dx + (3 * x + 3) * Dx ** 2 + (x ** 2 + 2 * x + 1) * Dx ** 3, x, 0, [1, 0, 2])
    assert p == q
    p = expr_to_holonomic(erf(x) ** 2 + x)
    q = HolonomicFunction((8 * x ** 4 - 2 * x ** 2 + 2) * Dx ** 2 + (6 * x ** 3 - x / 2) * Dx ** 3 + (x ** 2 + Rational(1, 4)) * Dx ** 4, x, 0, [0, 1, 8 / pi, 0])
    assert p == q
    p = expr_to_holonomic(cosh(x) * x)
    q = HolonomicFunction(-x ** 2 + 2 - 2 * x * Dx + x ** 2 * Dx ** 2, x, 0, [0, 1])
    assert p == q
    p = expr_to_holonomic(besselj(2, x))
    q = HolonomicFunction(x ** 2 - 4 + x * Dx + x ** 2 * Dx ** 2, x, 0, [0, 0])
    assert p == q
    p = expr_to_holonomic(besselj(0, x) + exp(x))
    q = HolonomicFunction(-x ** 2 - x / 2 + S.Half + (x ** 2 - x / 2 - Rational(3, 2)) * Dx + (-x ** 2 + x / 2 + 1) * Dx ** 2 + (x ** 2 + x / 2) * Dx ** 3, x, 0, [2, 1, S.Half])
    assert p == q
    p = expr_to_holonomic(sin(x) ** 2 / x)
    q = HolonomicFunction(4 + 4 * x * Dx + 3 * Dx ** 2 + x * Dx ** 3, x, 0, [0, 1, 0])
    assert p == q
    p = expr_to_holonomic(sin(x) ** 2 / x, x0=2)
    q = HolonomicFunction(4 + 4 * x * Dx + 3 * Dx ** 2 + x * Dx ** 3, x, 2, [sin(2) ** 2 / 2, sin(2) * cos(2) - sin(2) ** 2 / 4, -3 * sin(2) ** 2 / 4 + cos(2) ** 2 - sin(2) * cos(2)])
    assert p == q
    p = expr_to_holonomic(log(x) / 2 - Ci(2 * x) / 2 + Ci(2) / 2)
    q = HolonomicFunction(4 * Dx + 4 * x * Dx ** 2 + 3 * Dx ** 3 + x * Dx ** 4, x, 0, [-log(2) / 2 - EulerGamma / 2 + Ci(2) / 2, 0, 1, 0])
    assert p == q
    p = p.to_expr()
    q = log(x) / 2 - Ci(2 * x) / 2 + Ci(2) / 2
    assert p == q
    p = expr_to_holonomic(x ** S.Half, x0=1)
    q = HolonomicFunction(x * Dx - S.Half, x, 1, [1])
    assert p == q
    p = expr_to_holonomic(sqrt(1 + x ** 2))
    q = HolonomicFunction(-x + (x ** 2 + 1) * Dx, x, 0, [1])
    assert p == q
    assert (expr_to_holonomic(sqrt(x) + sqrt(2 * x)).to_expr() - (sqrt(x) + sqrt(2 * x))).simplify() == 0
    assert expr_to_holonomic(3 * x + 2 * sqrt(x)).to_expr() == 3 * x + 2 * sqrt(x)
    p = expr_to_holonomic((x ** 4 + x ** 3 + 5 * x ** 2 + 3 * x + 2) / x ** 2, lenics=3)
    q = HolonomicFunction(-2 * x ** 4 - x ** 3 + 3 * x + 4 + (x ** 5 + x ** 4 + 5 * x ** 3 + 3 * x ** 2 + 2 * x) * Dx, x, 0, {-2: [2, 3, 5]})
    assert p == q
    p = expr_to_holonomic(1 / (x - 1) ** 2, lenics=3, x0=1)
    q = HolonomicFunction(2 + (x - 1) * Dx, x, 1, {-2: [1, 0, 0]})
    assert p == q
    a = symbols('a')
    p = expr_to_holonomic(sqrt(a * x), x=x)
    assert p.to_expr() == sqrt(a) * sqrt(x)