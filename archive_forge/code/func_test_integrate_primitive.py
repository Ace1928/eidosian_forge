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
def test_integrate_primitive():
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1 / x, t)], 'Tfuncs': [log]})
    assert integrate_primitive(Poly(t, t), Poly(1, t), DE) == (x * log(x), -1, True)
    assert integrate_primitive(Poly(x, t), Poly(t, t), DE) == (0, NonElementaryIntegral(x / log(x), x), False)
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1 / x, t1), Poly(1 / (x + 1), t2)], 'Tfuncs': [log, Lambda(i, log(i + 1))]})
    assert integrate_primitive(Poly(t1, t2), Poly(t2, t2), DE) == (0, NonElementaryIntegral(log(x) / log(1 + x), x), False)
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1 / x, t1), Poly(1 / (x * t1), t2)], 'Tfuncs': [log, Lambda(i, log(log(i)))]})
    assert integrate_primitive(Poly(t2, t2), Poly(t1, t2), DE) == (0, NonElementaryIntegral(log(log(x)) / log(x), x), False)
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1 / x, t0)], 'Tfuncs': [log]})
    assert integrate_primitive(Poly(x ** 2 * t0 ** 3 + (3 * x ** 2 + x) * t0 ** 2 + (3 * x ** 2 + 2 * x) * t0 + x ** 2 + x, t0), Poly(x ** 2 * t0 ** 4 + 4 * x ** 2 * t0 ** 3 + 6 * x ** 2 * t0 ** 2 + 4 * x ** 2 * t0 + x ** 2, t0), DE) == (-1 / (log(x) + 1), NonElementaryIntegral(1 / (log(x) + 1), x), False)