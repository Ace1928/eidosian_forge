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
def test_canonical_representation():
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1 + t ** 2, t)]})
    assert canonical_representation(Poly(x - t, t), Poly(t ** 2, t), DE) == (Poly(0, t, domain='ZZ[x]'), (Poly(0, t, domain='QQ[x]'), Poly(1, t, domain='ZZ')), (Poly(-t + x, t, domain='QQ[x]'), Poly(t ** 2, t)))
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t ** 2 + 1, t)]})
    assert canonical_representation(Poly(t ** 5 + t ** 3 + x ** 2 * t + 1, t), Poly((t ** 2 + 1) ** 3, t), DE) == (Poly(0, t, domain='ZZ[x]'), (Poly(t ** 5 + t ** 3 + x ** 2 * t + 1, t, domain='QQ[x]'), Poly(t ** 6 + 3 * t ** 4 + 3 * t ** 2 + 1, t, domain='QQ')), (Poly(0, t, domain='QQ[x]'), Poly(1, t, domain='QQ')))