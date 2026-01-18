from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.function import (Derivative, Function)
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (acosh, asech)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, asin, atan, cos, sin)
from sympy.functions.special.bessel import airyai
from sympy.functions.special.error_functions import erf
from sympy.functions.special.gamma_functions import gamma
from sympy.integrals.integrals import integrate
from sympy.series.formal import fps
from sympy.series.order import O
from sympy.series.formal import (rational_algorithm, FormalPowerSeries,
from sympy.testing.pytest import raises, XFAIL, slow
def test_fps__product():
    f1, f2, f3 = (fps(sin(x)), fps(exp(x)), fps(cos(x)))
    raises(ValueError, lambda: f1.product(exp(x), x))
    raises(ValueError, lambda: f1.product(fps(exp(x), dir=-1), x, 4))
    raises(ValueError, lambda: f1.product(fps(exp(x), x0=1), x, 4))
    raises(ValueError, lambda: f1.product(fps(exp(y)), x, 4))
    fprod = f1.product(f2, x)
    assert isinstance(fprod, FormalPowerSeriesProduct)
    assert isinstance(fprod.ffps, FormalPowerSeries)
    assert isinstance(fprod.gfps, FormalPowerSeries)
    assert fprod.f == sin(x)
    assert fprod.g == exp(x)
    assert fprod.function == sin(x) * exp(x)
    assert fprod._eval_terms(4) == x + x ** 2 + x ** 3 / 3
    assert fprod.truncate(4) == x + x ** 2 + x ** 3 / 3 + O(x ** 4)
    assert fprod.polynomial(4) == x + x ** 2 + x ** 3 / 3
    raises(NotImplementedError, lambda: fprod._eval_term(5))
    raises(NotImplementedError, lambda: fprod.infinite)
    raises(NotImplementedError, lambda: fprod._eval_derivative(x))
    raises(NotImplementedError, lambda: fprod.integrate(x))
    assert f1.product(f3, x)._eval_terms(4) == x - 2 * x ** 3 / 3
    assert f1.product(f3, x).truncate(4) == x - 2 * x ** 3 / 3 + O(x ** 4)