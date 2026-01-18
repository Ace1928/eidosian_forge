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
def test_issue_19453():
    beta = Symbol('beta', positive=True)
    h = Symbol('h', positive=True)
    m = Symbol('m', positive=True)
    w = Symbol('omega', positive=True)
    g = Symbol('g', positive=True)
    e = exp(1)
    q = 3 * h ** 2 * beta * g * e ** (0.5 * h * beta * w)
    p = m ** 2 * w ** 2
    s = e ** (h * beta * w) - 1
    Z = -q / (4 * p * s) - q / (2 * p * s ** 2) - q * (e ** (h * beta * w) + 1) / (2 * p * s ** 3) + e ** (0.5 * h * beta * w) / s
    E = -diff(log(Z), beta)
    assert limit(E - 0.5 * h * w, beta, oo) == 0
    assert limit(E.simplify() - 0.5 * h * w, beta, oo) == 0