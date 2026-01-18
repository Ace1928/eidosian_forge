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
def test_manualintegrate_parts():
    assert manualintegrate(exp(x) * sin(x), x) == exp(x) * sin(x) / 2 - exp(x) * cos(x) / 2
    assert manualintegrate(2 * x * cos(x), x) == 2 * x * sin(x) + 2 * cos(x)
    assert manualintegrate(x * log(x), x) == x ** 2 * log(x) / 2 - x ** 2 / 4
    assert manualintegrate(log(x), x) == x * log(x) - x
    assert manualintegrate((3 * x ** 2 + 5) * exp(x), x) == 3 * x ** 2 * exp(x) - 6 * x * exp(x) + 11 * exp(x)
    assert manualintegrate(atan(x), x) == x * atan(x) - log(x ** 2 + 1) / 2
    assert _parts_rule(cos(x), x) == None
    assert _parts_rule(exp(x), x) == None
    assert _parts_rule(x ** 2, x) == None
    result = _parts_rule(atan(x), x)
    assert result[0] == atan(x) and result[1] == 1