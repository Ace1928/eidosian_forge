from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.function import (Derivative, Function, diff, expand)
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.relational import Ne
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (asinh, csch, cosh, coth, sech, sinh, tanh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (acos, acot, acsc, asec, asin, atan, cos, cot, csc, sec, sin, tan)
from sympy.functions.special.delta_functions import Heaviside, DiracDelta
from sympy.functions.special.elliptic_integrals import (elliptic_e, elliptic_f)
from sympy.functions.special.error_functions import (Chi, Ci, Ei, Shi, Si, erf, erfi, fresnelc, fresnels, li)
from sympy.functions.special.gamma_functions import uppergamma
from sympy.functions.special.polynomials import (assoc_laguerre, chebyshevt, chebyshevu, gegenbauer, hermite, jacobi, laguerre, legendre)
from sympy.functions.special.zeta_functions import polylog
from sympy.integrals.integrals import (Integral, integrate)
from sympy.logic.boolalg import And
from sympy.integrals.manualintegrate import (manualintegrate, find_substitutions,
from sympy.testing.pytest import raises, slow
def test_manualintegrate_trig_substitution():
    assert manualintegrate(sqrt(16 * x ** 2 - 9) / x, x) == Piecewise((sqrt(16 * x ** 2 - 9) - 3 * acos(3 / (4 * x)), And(x < Rational(3, 4), x > Rational(-3, 4))))
    assert manualintegrate(1 / (x ** 4 * sqrt(25 - x ** 2)), x) == Piecewise((-sqrt(-x ** 2 / 25 + 1) / (125 * x) - (-x ** 2 / 25 + 1) ** (3 * S.Half) / (15 * x ** 3), And(x < 5, x > -5)))
    assert manualintegrate(x ** 7 / (49 * x ** 2 + 1) ** (3 * S.Half), x) == (49 * x ** 2 + 1) ** (5 * S.Half) / 28824005 - (49 * x ** 2 + 1) ** (3 * S.Half) / 5764801 + 3 * sqrt(49 * x ** 2 + 1) / 5764801 + 1 / (5764801 * sqrt(49 * x ** 2 + 1))