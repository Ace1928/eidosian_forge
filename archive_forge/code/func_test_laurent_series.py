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
def test_laurent_series():
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1, t)]})
    a = Poly(36, t)
    d = Poly((t - 2) * (t ** 2 - 1) ** 2, t)
    F = Poly(t ** 2 - 1, t)
    n = 2
    assert laurent_series(a, d, F, n, DE) == (Poly(-3 * t ** 3 + 3 * t ** 2 - 6 * t - 8, t), Poly(t ** 5 + t ** 4 - 2 * t ** 3 - 2 * t ** 2 + t + 1, t), [Poly(-3 * t ** 3 - 6 * t ** 2, t, domain='QQ'), Poly(2 * t ** 6 + 6 * t ** 5 - 8 * t ** 3, t, domain='QQ')])