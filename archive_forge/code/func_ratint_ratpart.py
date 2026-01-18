from sympy.core.function import Lambda
from sympy.core.numbers import I
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.trigonometric import atan
from sympy.polys.polyroots import roots
from sympy.polys.polytools import cancel
from sympy.polys.rootoftools import RootSum
from sympy.polys import Poly, resultant, ZZ
def ratint_ratpart(f, g, x):
    """
    Horowitz-Ostrogradsky algorithm.

    Explanation
    ===========

    Given a field K and polynomials f and g in K[x], such that f and g
    are coprime and deg(f) < deg(g), returns fractions A and B in K(x),
    such that f/g = A' + B and B has square-free denominator.

    Examples
    ========

        >>> from sympy.integrals.rationaltools import ratint_ratpart
        >>> from sympy.abc import x, y
        >>> from sympy import Poly
        >>> ratint_ratpart(Poly(1, x, domain='ZZ'),
        ... Poly(x + 1, x, domain='ZZ'), x)
        (0, 1/(x + 1))
        >>> ratint_ratpart(Poly(1, x, domain='EX'),
        ... Poly(x**2 + y**2, x, domain='EX'), x)
        (0, 1/(x**2 + y**2))
        >>> ratint_ratpart(Poly(36, x, domain='ZZ'),
        ... Poly(x**5 - 2*x**4 - 2*x**3 + 4*x**2 + x - 2, x, domain='ZZ'), x)
        ((12*x + 6)/(x**2 - 1), 12/(x**2 - x - 2))

    See Also
    ========

    ratint, ratint_logpart
    """
    from sympy.solvers.solvers import solve
    f = Poly(f, x)
    g = Poly(g, x)
    u, v, _ = g.cofactors(g.diff())
    n = u.degree()
    m = v.degree()
    A_coeffs = [Dummy('a' + str(n - i)) for i in range(0, n)]
    B_coeffs = [Dummy('b' + str(m - i)) for i in range(0, m)]
    C_coeffs = A_coeffs + B_coeffs
    A = Poly(A_coeffs, x, domain=ZZ[C_coeffs])
    B = Poly(B_coeffs, x, domain=ZZ[C_coeffs])
    H = f - A.diff() * v + A * (u.diff() * v).quo(u) - B * u
    result = solve(H.coeffs(), C_coeffs)
    A = A.as_expr().subs(result)
    B = B.as_expr().subs(result)
    rat_part = cancel(A / u.as_expr(), x)
    log_part = cancel(B / v.as_expr(), x)
    return (rat_part, log_part)