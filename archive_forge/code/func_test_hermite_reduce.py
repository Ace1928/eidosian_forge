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
def test_hermite_reduce():
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t ** 2 + 1, t)]})
    assert hermite_reduce(Poly(x - t, t), Poly(t ** 2, t), DE) == ((Poly(-x, t, domain='QQ[x]'), Poly(t, t, domain='QQ[x]')), (Poly(0, t, domain='QQ[x]'), Poly(1, t, domain='QQ[x]')), (Poly(-x, t, domain='QQ[x]'), Poly(1, t, domain='QQ[x]')))
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(-t ** 2 - t / x - (1 - nu ** 2 / x ** 2), t)]})
    assert hermite_reduce(Poly(x ** 2 * t ** 5 + x * t ** 4 - nu ** 2 * t ** 3 - x * (x ** 2 + 1) * t ** 2 - (x ** 2 - nu ** 2) * t - x ** 5 / 4, t), Poly(x ** 2 * t ** 4 + x ** 2 * (x ** 2 + 2) * t ** 2 + x ** 2 + x ** 4 + x ** 6 / 4, t), DE) == ((Poly(-x ** 2 - 4, t, domain='ZZ(x,nu)'), Poly(4 * t ** 2 + 2 * x ** 2 + 4, t, domain='ZZ(x,nu)')), (Poly((-2 * nu ** 2 - x ** 4) * t - (2 * x ** 3 + 2 * x), t, domain='ZZ(x,nu)'), Poly(2 * x ** 2 * t ** 2 + x ** 4 + 2 * x ** 2, t, domain='ZZ(x,nu)')), (Poly(x * t + 1, t, domain='ZZ(x,nu)'), Poly(x, t, domain='ZZ(x,nu)')))
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1 / x, t)]})
    a = Poly((-2 + 3 * x) * t ** 3 + (-1 + x) * t ** 2 + (-4 * x + 2 * x ** 2) * t + x ** 2, t)
    d = Poly(x * t ** 6 - 4 * x ** 2 * t ** 5 + 6 * x ** 3 * t ** 4 - 4 * x ** 4 * t ** 3 + x ** 5 * t ** 2, t)
    assert hermite_reduce(a, d, DE) == ((Poly(3 * t ** 2 + t + 3 * x, t, domain='ZZ(x)'), Poly(3 * t ** 4 - 9 * x * t ** 3 + 9 * x ** 2 * t ** 2 - 3 * x ** 3 * t, t, domain='ZZ(x)')), (Poly(0, t, domain='ZZ(x)'), Poly(1, t, domain='ZZ(x)')), (Poly(0, t, domain='ZZ(x)'), Poly(1, t, domain='ZZ(x)')))
    assert hermite_reduce(Poly(-t ** 2 + 2 * t + 2, t, domain='ZZ(x)'), Poly(-x * t ** 2 + 2 * x * t - x, t, domain='ZZ(x)'), DE) == ((Poly(3, t, domain='ZZ(x)'), Poly(t - 1, t, domain='ZZ(x)')), (Poly(0, t, domain='ZZ(x)'), Poly(1, t, domain='ZZ(x)')), (Poly(1, t, domain='ZZ(x)'), Poly(x, t, domain='ZZ(x)')))
    assert hermite_reduce(Poly(-x ** 2 * t ** 6 + (-1 - 2 * x ** 3 + x ** 4) * t ** 3 + (-3 - 3 * x ** 4) * t ** 2 - 2 * x * t - x - 3 * x ** 2, t, domain='ZZ(x)'), Poly(x ** 4 * t ** 6 - 2 * x ** 2 * t ** 3 + 1, t, domain='ZZ(x)'), DE) == ((Poly(x ** 3 * t + x ** 4 + 1, t, domain='ZZ(x)'), Poly(x ** 3 * t ** 3 - x, t, domain='ZZ(x)')), (Poly(0, t, domain='ZZ(x)'), Poly(1, t, domain='ZZ(x)')), (Poly(-1, t, domain='ZZ(x)'), Poly(x ** 2, t, domain='ZZ(x)')))
    assert hermite_reduce(Poly((-2 + 3 * x) * t ** 3 + (-1 + x) * t ** 2 + (-4 * x + 2 * x ** 2) * t + x ** 2, t), Poly(x * t ** 6 - 4 * x ** 2 * t ** 5 + 6 * x ** 3 * t ** 4 - 4 * x ** 4 * t ** 3 + x ** 5 * t ** 2, t), DE) == ((Poly(3 * t ** 2 + t + 3 * x, t, domain='ZZ(x)'), Poly(3 * t ** 4 - 9 * x * t ** 3 + 9 * x ** 2 * t ** 2 - 3 * x ** 3 * t, t, domain='ZZ(x)')), (Poly(0, t, domain='ZZ(x)'), Poly(1, t, domain='ZZ(x)')), (Poly(0, t, domain='ZZ(x)'), Poly(1, t, domain='ZZ(x)')))