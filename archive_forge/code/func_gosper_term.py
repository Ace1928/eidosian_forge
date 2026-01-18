from sympy.core import S, Dummy, symbols
from sympy.polys import Poly, parallel_poly_from_expr, factor
from sympy.utilities.iterables import is_sequence
def gosper_term(f, n):
    """
    Compute Gosper's hypergeometric term for ``f``.

    Explanation
    ===========

    Suppose ``f`` is a hypergeometric term such that:

    .. math::
        s_n = \\sum_{k=0}^{n-1} f_k

    and `f_k` does not depend on `n`. Returns a hypergeometric
    term `g_n` such that `g_{n+1} - g_n = f_n`.

    Examples
    ========

    >>> from sympy.concrete.gosper import gosper_term
    >>> from sympy import factorial
    >>> from sympy.abc import n

    >>> gosper_term((4*n + 1)*factorial(n)/factorial(2*n + 1), n)
    (-n - 1/2)/(n + 1/4)

    """
    from sympy.simplify import hypersimp
    r = hypersimp(f, n)
    if r is None:
        return None
    p, q = r.as_numer_denom()
    A, B, C = gosper_normal(p, q, n)
    B = B.shift(-1)
    N = S(A.degree())
    M = S(B.degree())
    K = S(C.degree())
    if N != M or A.LC() != B.LC():
        D = {K - max(N, M)}
    elif not N:
        D = {K - N + 1, S.Zero}
    else:
        D = {K - N + 1, (B.nth(N - 1) - A.nth(N - 1)) / A.LC()}
    for d in set(D):
        if not d.is_Integer or d < 0:
            D.remove(d)
    if not D:
        return None
    d = max(D)
    coeffs = symbols('c:%s' % (d + 1), cls=Dummy)
    domain = A.get_domain().inject(*coeffs)
    x = Poly(coeffs, n, domain=domain)
    H = A * x.shift(1) - B * x - C
    from sympy.solvers.solvers import solve
    solution = solve(H.coeffs(), coeffs)
    if solution is None:
        return None
    x = x.as_expr().subs(solution)
    for coeff in coeffs:
        if coeff not in solution:
            x = x.subs(coeff, 0)
    if x.is_zero:
        return None
    else:
        return B.as_expr() * x / C.as_expr()