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
def test_integrate_Abs_sign():
    assert integrate(Abs(x), (x, -2, 1)) == Rational(5, 2)
    assert integrate(Abs(x), (x, 0, 1)) == S.Half
    assert integrate(Abs(x + 1), (x, 0, 1)) == Rational(3, 2)
    assert integrate(Abs(x ** 2 - 1), (x, -2, 2)) == 4
    assert integrate(Abs(x ** 2 - 3 * x), (x, -15, 15)) == 2259
    assert integrate(sign(x), (x, -1, 2)) == 1
    assert integrate(sign(x) * sin(x), (x, -pi, pi)) == 4
    assert integrate(sign(x - 2) * x ** 2, (x, 0, 3)) == Rational(11, 3)
    t, s = symbols('t s', real=True)
    assert integrate(Abs(t), t) == Piecewise((-t ** 2 / 2, t <= 0), (t ** 2 / 2, True))
    assert integrate(Abs(2 * t - 6), t) == Piecewise((-t ** 2 + 6 * t, t <= 3), (t ** 2 - 6 * t + 18, True))
    assert integrate(abs(t - s ** 2), (t, 0, 2)) == 2 * s ** 2 * Min(2, s ** 2) - 2 * s ** 2 - Min(2, s ** 2) ** 2 + 2
    assert integrate(exp(-Abs(t)), t) == Piecewise((exp(t), t <= 0), (2 - exp(-t), True))
    assert integrate(sign(2 * t - 6), t) == Piecewise((-t, t < 3), (t - 6, True))
    assert integrate(2 * t * sign(t ** 2 - 1), t) == Piecewise((t ** 2, t < -1), (-t ** 2 + 2, t < 1), (t ** 2, True))
    assert integrate(sign(t), (t, s + 1)) == Piecewise((s + 1, s + 1 > 0), (-s - 1, s + 1 < 0), (0, True))