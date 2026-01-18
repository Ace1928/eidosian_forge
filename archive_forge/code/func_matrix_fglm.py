from sympy.polys.monomials import monomial_mul, monomial_div
def matrix_fglm(F, ring, O_to):
    """
    Converts the reduced Groebner basis ``F`` of a zero-dimensional
    ideal w.r.t. ``O_from`` to a reduced Groebner basis
    w.r.t. ``O_to``.

    References
    ==========

    .. [1] J.C. Faugere, P. Gianni, D. Lazard, T. Mora (1994). Efficient
           Computation of Zero-dimensional Groebner Bases by Change of
           Ordering
    """
    domain = ring.domain
    ngens = ring.ngens
    ring_to = ring.clone(order=O_to)
    old_basis = _basis(F, ring)
    M = _representing_matrices(old_basis, F, ring)
    S = [ring.zero_monom]
    V = [[domain.one] + [domain.zero] * (len(old_basis) - 1)]
    G = []
    L = [(i, 0) for i in range(ngens)]
    L.sort(key=lambda k_l: O_to(_incr_k(S[k_l[1]], k_l[0])), reverse=True)
    t = L.pop()
    P = _identity_matrix(len(old_basis), domain)
    while True:
        s = len(S)
        v = _matrix_mul(M[t[0]], V[t[1]])
        _lambda = _matrix_mul(P, v)
        if all((_lambda[i] == domain.zero for i in range(s, len(old_basis)))):
            lt = ring.term_new(_incr_k(S[t[1]], t[0]), domain.one)
            rest = ring.from_dict({S[i]: _lambda[i] for i in range(s)})
            g = (lt - rest).set_ring(ring_to)
            if g:
                G.append(g)
        else:
            P = _update(s, _lambda, P)
            S.append(_incr_k(S[t[1]], t[0]))
            V.append(v)
            L.extend([(i, s) for i in range(ngens)])
            L = list(set(L))
            L.sort(key=lambda k_l: O_to(_incr_k(S[k_l[1]], k_l[0])), reverse=True)
        L = [(k, l) for k, l in L if all((monomial_div(_incr_k(S[l], k), g.LM) is None for g in G))]
        if not L:
            G = [g.monic() for g in G]
            return sorted(G, key=lambda g: O_to(g.LM), reverse=True)
        t = L.pop()