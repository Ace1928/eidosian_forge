from sympy.core.function import (Function, Lambda, diff, expand_log)
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.relational import Ne
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (atan, sin, tan)
from sympy.polys.polytools import (Poly, cancel, factor)
from sympy.integrals.risch import (gcdex_diophantine, frac_in, as_poly_1t,
from sympy.testing.pytest import raises
from sympy.abc import x, t, nu, z, a, y
def test_integrate_nonlinear_no_specials():
    a, d = (Poly(x ** 2 * t ** 5 + x * t ** 4 - nu ** 2 * t ** 3 - x * (x ** 2 + 1) * t ** 2 - (x ** 2 - nu ** 2) * t - x ** 5 / 4, t), Poly(x ** 2 * t ** 4 + x ** 2 * (x ** 2 + 2) * t ** 2 + x ** 2 + x ** 4 + x ** 6 / 4, t))
    f = Function('phi_nu')
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(-t ** 2 - t / x - (1 - nu ** 2 / x ** 2), t)], 'Tfuncs': [f]})
    assert integrate_nonlinear_no_specials(a, d, DE) == (-log(1 + f(x) ** 2 + x ** 2 / 2) / 2 + (-4 - x ** 2) / (4 + 2 * x ** 2 + 4 * f(x) ** 2), True)
    assert integrate_nonlinear_no_specials(Poly(t, t), Poly(1, t), DE) == (0, False)