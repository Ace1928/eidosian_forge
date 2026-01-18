from itertools import product
from sympy.concrete.summations import Sum
from sympy.core.function import (Function, diff)
from sympy.core import EulerGamma
from sympy.core.numbers import (E, I, Rational, oo, pi, zoo)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.combinatorial.factorials import (binomial, factorial, subfactorial)
from sympy.functions.elementary.complexes import (Abs, re, sign)
from sympy.functions.elementary.exponential import (LambertW, exp, log)
from sympy.functions.elementary.hyperbolic import (acosh, acoth, acsch, asech, atanh, sinh, tanh)
from sympy.functions.elementary.integers import (ceiling, floor, frac)
from sympy.functions.elementary.miscellaneous import (cbrt, real_root, sqrt)
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (acos, acot, acsc, asec, asin,
from sympy.functions.special.bessel import (besseli, bessely, besselj, besselk)
from sympy.functions.special.error_functions import (Ei, erf, erfc, erfi, fresnelc, fresnels)
from sympy.functions.special.gamma_functions import (digamma, gamma, uppergamma)
from sympy.functions.special.hyper import meijerg
from sympy.integrals.integrals import (Integral, integrate)
from sympy.series.limits import (Limit, limit)
from sympy.simplify.simplify import (logcombine, simplify)
from sympy.simplify.hyperexpand import hyperexpand
from sympy.calculus.accumulationbounds import AccumBounds
from sympy.core.mul import Mul
from sympy.series.limits import heuristics
from sympy.series.order import Order
from sympy.testing.pytest import XFAIL, raises
from sympy.abc import x, y, z, k
def test_issue_22220():
    e1 = sqrt(30) * atan(sqrt(30) * tan(x / 2) / 6) / 30
    e2 = sqrt(30) * I * (-log(sqrt(2) * tan(x / 2) - 2 * sqrt(15) * I / 5) + +log(sqrt(2) * tan(x / 2) + 2 * sqrt(15) * I / 5)) / 60
    assert limit(e1, x, -pi) == -sqrt(30) * pi / 60
    assert limit(e2, x, -pi) == -sqrt(30) * pi / 30
    assert limit(e1, x, -pi, '-') == sqrt(30) * pi / 60
    assert limit(e2, x, -pi, '-') == 0
    expr = log(x - I) - log(-x - I)
    expr2 = logcombine(expr, force=True)
    assert limit(expr, x, oo) == limit(expr2, x, oo) == I * pi
    expr = expr = -log(tan(x / 2) - I) + log(tan(x / 2) + I)
    assert limit(expr, x, pi, '+') == 2 * I * pi
    assert limit(expr, x, pi, '-') == 0