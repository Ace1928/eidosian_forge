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
def test_NonElementaryIntegral():
    assert isinstance(risch_integrate(exp(x ** 2), x), NonElementaryIntegral)
    assert isinstance(risch_integrate(x ** x * log(x), x), NonElementaryIntegral)
    assert isinstance(NonElementaryIntegral(x ** x * t0, x).subs(t0, log(x)), NonElementaryIntegral)