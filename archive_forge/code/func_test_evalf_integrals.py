import math
from sympy.concrete.summations import (Sum, summation)
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import (Derivative, Function, Lambda, diff)
from sympy.core import EulerGamma
from sympy.core.numbers import (E, Float, I, Rational, nan, oo, pi, zoo)
from sympy.core.relational import (Eq, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.functions.elementary.complexes import (Abs, im, polar_lift, re, sign)
from sympy.functions.elementary.exponential import (LambertW, exp, exp_polar, log)
from sympy.functions.elementary.hyperbolic import (acosh, asinh, cosh, coth, csch, sinh, tanh, sech)
from sympy.functions.elementary.miscellaneous import (Max, Min, sqrt)
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (acos, asin, atan, cos, sin, sinc, tan, sec)
from sympy.functions.special.delta_functions import DiracDelta, Heaviside
from sympy.functions.special.error_functions import (Ci, Ei, Si, erf, erfc, erfi, fresnelc, li)
from sympy.functions.special.gamma_functions import (gamma, polygamma)
from sympy.functions.special.hyper import (hyper, meijerg)
from sympy.functions.special.singularity_functions import SingularityFunction
from sympy.functions.special.zeta_functions import lerchphi
from sympy.integrals.integrals import integrate
from sympy.logic.boolalg import And
from sympy.matrices.dense import Matrix
from sympy.polys.polytools import (Poly, factor)
from sympy.printing.str import sstr
from sympy.series.order import O
from sympy.sets.sets import Interval
from sympy.simplify.gammasimp import gammasimp
from sympy.simplify.simplify import simplify
from sympy.simplify.trigsimp import trigsimp
from sympy.tensor.indexed import (Idx, IndexedBase)
from sympy.core.expr import unchanged
from sympy.functions.elementary.integers import floor
from sympy.integrals.integrals import Integral
from sympy.integrals.risch import NonElementaryIntegral
from sympy.physics import units
from sympy.testing.pytest import (raises, slow, skip, ON_CI,
from sympy.utilities.exceptions import SymPyDeprecationWarning
from sympy.core.random import verify_numerically
@slow
def test_evalf_integrals():
    assert NS(Integral(x, (x, 2, 5)), 15) == '10.5000000000000'
    gauss = Integral(exp(-x ** 2), (x, -oo, oo))
    assert NS(gauss, 15) == '1.77245385090552'
    assert NS(gauss ** 2 - pi + E * Rational(1, 10 ** 20), 15) in ('2.71828182845904e-20', '2.71828182845905e-20')
    t = Symbol('t')
    a = 8 * sqrt(3) / (1 + 3 * t ** 2)
    b = 16 * sqrt(2) * (3 * t + 1) * sqrt(4 * t ** 2 + t + 1) ** 3
    c = (3 * t ** 2 + 1) * (11 * t ** 2 + 2 * t + 3) ** 2
    d = sqrt(2) * (249 * t ** 2 + 54 * t + 65) / (11 * t ** 2 + 2 * t + 3) ** 2
    f = a - b / c - d
    assert NS(Integral(f, (t, 0, 1)), 50) == NS((3 * sqrt(2) - 49 * pi + 162 * atan(sqrt(2))) / 12, 50)
    assert NS(Integral(log(log(1 / x)) / (1 + x + x ** 2), (x, 0, 1)), 15) == NS('pi/sqrt(3) * log(2*pi**(5/6) / gamma(1/6))', 15)
    assert NS(Integral(atan(sqrt(x ** 2 + 2)) / (sqrt(x ** 2 + 2) * (x ** 2 + 1)), (x, 0, 1)), 15) == NS(5 * pi ** 2 / 96, 15)
    assert NS(Integral(x / ((exp(pi * x) - exp(-pi * x)) * (x ** 2 + 1)), (x, 0, oo)), 15) == NS('log(2)/2-1/4', 15)
    assert NS(Integral(log(log(sin(x) / cos(x))), (x, pi / 4, pi / 2)), 15, chop=True) == NS('pi/4*log(4*pi**3/gamma(1/4)**4)', 15)
    assert NS(2 + Integral(log(2 * cos(x / 2)), (x, -pi, pi)), 17, chop=True) == NS(2, 17)
    assert NS(2 + Integral(log(2 * cos(x / 2)), (x, -pi, pi)), 20, chop=True) == NS(2, 20)
    assert NS(2 + Integral(log(2 * cos(x / 2)), (x, -pi, pi)), 22, chop=True) == NS(2, 22)
    assert NS(pi - 4 * Integral('sqrt(1-x**2)', (x, 0, 1)), 15, maxn=30, chop=True) in ('0.0', '0')
    a = Integral(sin(x) / x ** 2, (x, 1, oo)).evalf(maxn=15)
    assert 0.49 < a < 0.51
    assert NS(Integral(sin(x) / x ** 2, (x, 1, oo)), quad='osc') == '0.504067061906928'
    assert NS(Integral(cos(pi * x + 1) / x, (x, -oo, -1)), quad='osc') == '0.276374705640365'
    assert NS(Integral(x, x)) == 'Integral(x, x)'
    assert NS(Integral(x, (x, y))) == 'Integral(x, (x, y))'