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
def rsolve_poly(coeffs, f, n, shift=0, **hints):
    """
    Given linear recurrence operator `\\operatorname{L}` of order
    `k` with polynomial coefficients and inhomogeneous equation
    `\\operatorname{L} y = f`, where `f` is a polynomial, we seek for
    all polynomial solutions over field `K` of characteristic zero.

    The algorithm performs two basic steps:

        (1) Compute degree `N` of the general polynomial solution.
        (2) Find all polynomials of degree `N` or less
            of `\\operatorname{L} y = f`.

    There are two methods for computing the polynomial solutions.
    If the degree bound is relatively small, i.e. it's smaller than
    or equal to the order of the recurrence, then naive method of
    undetermined coefficients is being used. This gives a system
    of algebraic equations with `N+1` unknowns.

    In the other case, the algorithm performs transformation of the
    initial equation to an equivalent one for which the system of
    algebraic equations has only `r` indeterminates. This method is
    quite sophisticated (in comparison with the naive one) and was
    invented together by Abramov, Bronstein and Petkovsek.

    It is possible to generalize the algorithm implemented here to
    the case of linear q-difference and differential equations.

    Lets say that we would like to compute `m`-th Bernoulli polynomial
    up to a constant. For this we can use `b(n+1) - b(n) = m n^{m-1}`
    recurrence, which has solution `b(n) = B_m + C`. For example:

    >>> from sympy import Symbol, rsolve_poly
    >>> n = Symbol('n', integer=True)

    >>> rsolve_poly([-1, 1], 4*n**3, n)
    C0 + n**4 - 2*n**3 + n**2

    References
    ==========

    .. [1] S. A. Abramov, M. Bronstein and M. Petkovsek, On polynomial
           solutions of linear operator equations, in: T. Levelt, ed.,
           Proc. ISSAC '95, ACM Press, New York, 1995, 290-296.

    .. [2] M. Petkovsek, Hypergeometric solutions of linear recurrences
           with polynomial coefficients, J. Symbolic Computation,
           14 (1992), 243-264.

    .. [3] M. Petkovsek, H. S. Wilf, D. Zeilberger, A = B, 1996.

    """
    f = sympify(f)
    if not f.is_polynomial(n):
        return None
    homogeneous = f.is_zero
    r = len(coeffs) - 1
    coeffs = [Poly(coeff, n) for coeff in coeffs]
    polys = [Poly(0, n)] * (r + 1)
    terms = [(S.Zero, S.NegativeInfinity)] * (r + 1)
    for i in range(r + 1):
        for j in range(i, r + 1):
            polys[i] += coeffs[j] * binomial(j, i).as_poly(n)
        if not polys[i].is_zero:
            (exp,), coeff = polys[i].LT()
            terms[i] = (coeff, exp)
    d = b = terms[0][1]
    for i in range(1, r + 1):
        if terms[i][1] > d:
            d = terms[i][1]
        if terms[i][1] - i > b:
            b = terms[i][1] - i
    d, b = (int(d), int(b))
    x = Dummy('x')
    degree_poly = S.Zero
    for i in range(r + 1):
        if terms[i][1] - i == b:
            degree_poly += terms[i][0] * FallingFactorial(x, i)
    nni_roots = list(roots(degree_poly, x, filter='Z', predicate=lambda r: r >= 0).keys())
    if nni_roots:
        N = [max(nni_roots)]
    else:
        N = []
    if homogeneous:
        N += [-b - 1]
    else:
        N += [f.as_poly(n).degree() - b, -b - 1]
    N = int(max(N))
    if N < 0:
        if homogeneous:
            if hints.get('symbols', False):
                return (S.Zero, [])
            else:
                return S.Zero
        else:
            return None
    if N <= r:
        C = []
        y = E = S.Zero
        for i in range(N + 1):
            C.append(Symbol('C' + str(i + shift)))
            y += C[i] * n ** i
        for i in range(r + 1):
            E += coeffs[i].as_expr() * y.subs(n, n + i)
        solutions = solve_undetermined_coeffs(E - f, C, n)
        if solutions is not None:
            _C = C
            C = [c for c in C if c not in solutions]
            result = y.subs(solutions)
        else:
            return None
    else:
        A = r
        U = N + A + b + 1
        nni_roots = list(roots(polys[r], filter='Z', predicate=lambda r: r >= 0).keys())
        if nni_roots != []:
            a = max(nni_roots) + 1
        else:
            a = S.Zero

        def _zero_vector(k):
            return [S.Zero] * k

        def _one_vector(k):
            return [S.One] * k

        def _delta(p, k):
            B = S.One
            D = p.subs(n, a + k)
            for i in range(1, k + 1):
                B *= Rational(i - k - 1, i)
                D += B * p.subs(n, a + k - i)
            return D
        alpha = {}
        for i in range(-A, d + 1):
            I = _one_vector(d + 1)
            for k in range(1, d + 1):
                I[k] = I[k - 1] * (x + i - k + 1) / k
            alpha[i] = S.Zero
            for j in range(A + 1):
                for k in range(d + 1):
                    B = binomial(k, i + j)
                    D = _delta(polys[j].as_expr(), k)
                    alpha[i] += I[k] * B * D
        V = Matrix(U, A, lambda i, j: int(i == j))
        if homogeneous:
            for i in range(A, U):
                v = _zero_vector(A)
                for k in range(1, A + b + 1):
                    if i - k < 0:
                        break
                    B = alpha[k - A].subs(x, i - k)
                    for j in range(A):
                        v[j] += B * V[i - k, j]
                denom = alpha[-A].subs(x, i)
                for j in range(A):
                    V[i, j] = -v[j] / denom
        else:
            G = _zero_vector(U)
            for i in range(A, U):
                v = _zero_vector(A)
                g = S.Zero
                for k in range(1, A + b + 1):
                    if i - k < 0:
                        break
                    B = alpha[k - A].subs(x, i - k)
                    for j in range(A):
                        v[j] += B * V[i - k, j]
                    g += B * G[i - k]
                denom = alpha[-A].subs(x, i)
                for j in range(A):
                    V[i, j] = -v[j] / denom
                G[i] = (_delta(f, i - A) - g) / denom
        P, Q = (_one_vector(U), _zero_vector(A))
        for i in range(1, U):
            P[i] = (P[i - 1] * (n - a - i + 1) / i).expand()
        for i in range(A):
            Q[i] = Add(*[(v * p).expand() for v, p in zip(V[:, i], P)])
        if not homogeneous:
            h = Add(*[(g * p).expand() for g, p in zip(G, P)])
        C = [Symbol('C' + str(i + shift)) for i in range(A)]
        g = lambda i: Add(*[c * _delta(q, i) for c, q in zip(C, Q)])
        if homogeneous:
            E = [g(i) for i in range(N + 1, U)]
        else:
            E = [g(i) + _delta(h, i) for i in range(N + 1, U)]
        if E != []:
            solutions = solve(E, *C)
            if not solutions:
                if homogeneous:
                    if hints.get('symbols', False):
                        return (S.Zero, [])
                    else:
                        return S.Zero
                else:
                    return None
        else:
            solutions = {}
        if homogeneous:
            result = S.Zero
        else:
            result = h
        _C = C[:]
        for c, q in list(zip(C, Q)):
            if c in solutions:
                s = solutions[c] * q
                C.remove(c)
            else:
                s = c * q
            result += s.expand()
    if C != _C:
        result = result.xreplace(dict(zip(C, _C)))
        C = _C[:len(C)]
    if hints.get('symbols', False):
        return (result, C)
    else:
        return result