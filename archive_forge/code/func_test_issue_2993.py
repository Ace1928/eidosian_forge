from sympy.core import (
from sympy.core.parameters import global_parameters
from sympy.core.tests.test_evalf import NS
from sympy.core.function import expand_multinomial
from sympy.functions.elementary.miscellaneous import sqrt, cbrt
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.special.error_functions import erf
from sympy.functions.elementary.trigonometric import (
from sympy.functions.elementary.hyperbolic import cosh, sinh, tanh
from sympy.polys import Poly
from sympy.series.order import O
from sympy.sets import FiniteSet
from sympy.core.power import power, integer_nthroot
from sympy.testing.pytest import warns, _both_exp_pow
from sympy.utilities.exceptions import SymPyDeprecationWarning
from sympy.abc import a, b, c, x, y
def test_issue_2993():
    assert str((2.3 * x - 4) ** 0.3) == '1.5157165665104*(0.575*x - 1)**0.3'
    assert str((2.3 * x + 4) ** 0.3) == '1.5157165665104*(0.575*x + 1)**0.3'
    assert str((-2.3 * x + 4) ** 0.3) == '1.5157165665104*(1 - 0.575*x)**0.3'
    assert str((-2.3 * x - 4) ** 0.3) == '1.5157165665104*(-0.575*x - 1)**0.3'
    assert str((2.3 * x - 2) ** 0.3) == '1.28386201800527*(x - 0.869565217391304)**0.3'
    assert str((-2.3 * x - 2) ** 0.3) == '1.28386201800527*(-x - 0.869565217391304)**0.3'
    assert str((-2.3 * x + 2) ** 0.3) == '1.28386201800527*(0.869565217391304 - x)**0.3'
    assert str((2.3 * x + 2) ** 0.3) == '1.28386201800527*(x + 0.869565217391304)**0.3'
    assert str((2.3 * x - 4) ** Rational(1, 3)) == '2**(2/3)*(0.575*x - 1)**(1/3)'
    eq = 2.3 * x + 4
    assert eq ** 2 == 16 * (0.575 * x + 1) ** 2
    assert (1 / eq).args == (eq, -1)
    q = 0.5 * exp(x) - 0.5 * exp(-x) + 0.1
    assert int((q ** 2).subs(x, 1)) == 1
    y = Symbol('y')
    assert len(sqrt(x / (x + y) ** 2 + Float('0.008', 30)).subs(y, pi.n(25)).atoms(Float)) == 2
    a, b, c, d, e, f, g = symbols('a:g')
    expr = sqrt(1 + a * (c ** 4 + g * d - 2 * g * e - f * (-g + d)) ** 2 / (c ** 3 * b ** 2 * (d - 3 * e + 2 * f) ** 2)) / 2
    r = [(a, N('0.0170992456333788667034850458615', 30)), (b, N('0.0966594956075474769169134801223', 30)), (c, N('0.390911862903463913632151616184', 30)), (d, N('0.152812084558656566271750185933', 30)), (e, N('0.137562344465103337106561623432', 30)), (f, N('0.174259178881496659302933610355', 30)), (g, N('0.220745448491223779615401870086', 30))]
    tru = expr.n(30, subs=dict(r))
    seq = expr.subs(r)
    assert seq == tru