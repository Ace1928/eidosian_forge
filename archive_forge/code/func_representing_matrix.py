from sympy.polys.monomials import monomial_mul, monomial_div
def representing_matrix(m):
    M = [[domain.zero] * len(basis) for _ in range(len(basis))]
    for i, v in enumerate(basis):
        r = ring.term_new(monomial_mul(m, v), domain.one).rem(G)
        for monom, coeff in r.terms():
            j = basis.index(monom)
            M[j][i] = coeff
    return M