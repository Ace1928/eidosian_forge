from collections import defaultdict
from sympy.concrete import product
from sympy.core.singleton import S
from sympy.core.numbers import Rational, I
from sympy.core.symbol import Symbol, Wild, Dummy
from sympy.core.relational import Equality
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.sorting import default_sort_key
from sympy.core.sympify import sympify
from sympy.simplify import simplify, hypersimp, hypersimilar  # type: ignore
from sympy.solvers import solve, solve_undetermined_coeffs
from sympy.polys import Poly, quo, gcd, lcm, roots, resultant
from sympy.functions import binomial, factorial, FallingFactorial, RisingFactorial
from sympy.matrices import Matrix, casoratian
from sympy.utilities.iterables import numbered_symbols
def rsolve_ratio(coeffs, f, n, **hints):
    """
    Given linear recurrence operator `\\operatorname{L}` of order `k`
    with polynomial coefficients and inhomogeneous equation
    `\\operatorname{L} y = f`, where `f` is a polynomial, we seek
    for all rational solutions over field `K` of characteristic zero.

    This procedure accepts only polynomials, however if you are
    interested in solving recurrence with rational coefficients
    then use ``rsolve`` which will pre-process the given equation
    and run this procedure with polynomial arguments.

    The algorithm performs two basic steps:

        (1) Compute polynomial `v(n)` which can be used as universal
            denominator of any rational solution of equation
            `\\operatorname{L} y = f`.

        (2) Construct new linear difference equation by substitution
            `y(n) = u(n)/v(n)` and solve it for `u(n)` finding all its
            polynomial solutions. Return ``None`` if none were found.

    The algorithm implemented here is a revised version of the original
    Abramov's algorithm, developed in 1989. The new approach is much
    simpler to implement and has better overall efficiency. This
    method can be easily adapted to the q-difference equations case.

    Besides finding rational solutions alone, this functions is
    an important part of Hyper algorithm where it is used to find
    a particular solution for the inhomogeneous part of a recurrence.

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy.solvers.recurr import rsolve_ratio
    >>> rsolve_ratio([-2*x**3 + x**2 + 2*x - 1, 2*x**3 + x**2 - 6*x,
    ... - 2*x**3 - 11*x**2 - 18*x - 9, 2*x**3 + 13*x**2 + 22*x + 8], 0, x)
    C0*(2*x - 3)/(2*(x**2 - 1))

    References
    ==========

    .. [1] S. A. Abramov, Rational solutions of linear difference
           and q-difference equations with polynomial coefficients,
           in: T. Levelt, ed., Proc. ISSAC '95, ACM Press, New York,
           1995, 285-289

    See Also
    ========

    rsolve_hyper
    """
    f = sympify(f)
    if not f.is_polynomial(n):
        return None
    coeffs = list(map(sympify, coeffs))
    r = len(coeffs) - 1
    A, B = (coeffs[r], coeffs[0])
    A = A.subs(n, n - r).expand()
    h = Dummy('h')
    res = resultant(A, B.subs(n, n + h), n)
    if not res.is_polynomial(h):
        p, q = res.as_numer_denom()
        res = quo(p, q, h)
    nni_roots = list(roots(res, h, filter='Z', predicate=lambda r: r >= 0).keys())
    if not nni_roots:
        return rsolve_poly(coeffs, f, n, **hints)
    else:
        C, numers = (S.One, [S.Zero] * (r + 1))
        for i in range(int(max(nni_roots)), -1, -1):
            d = gcd(A, B.subs(n, n + i), n)
            A = quo(A, d, n)
            B = quo(B, d.subs(n, n - i), n)
            C *= Mul(*[d.subs(n, n - j) for j in range(i + 1)])
        denoms = [C.subs(n, n + i) for i in range(r + 1)]
        for i in range(r + 1):
            g = gcd(coeffs[i], denoms[i], n)
            numers[i] = quo(coeffs[i], g, n)
            denoms[i] = quo(denoms[i], g, n)
        for i in range(r + 1):
            numers[i] *= Mul(*denoms[:i] + denoms[i + 1:])
        result = rsolve_poly(numers, f * Mul(*denoms), n, **hints)
        if result is not None:
            if hints.get('symbols', False):
                return (simplify(result[0] / C), result[1])
            else:
                return simplify(result / C)
        else:
            return None