import string
from ..sage_helper import _within_sage, sage_method
def compute_torsion(G, bits_prec, alpha=None, phi=None, phialpha=None, return_parts=False, return_as_poly=True, wada_conventions=False, symmetry_test=True):
    if alpha:
        F = alpha('a').base_ring()
    elif phialpha:
        F = phialpha('a').base_ring().base_ring()
    epsilon = ZZ(2) ** (-bits_prec // 3) if not F.is_exact() else None
    big_epsilon = ZZ(2) ** (-bits_prec // 5) if not F.is_exact() else None
    gens, rels = (G.generators(), G.relators())
    if len(rels) != len(gens) - 1:
        raise ValueError('Algorithm to compute torsion requires a group presentation with deficiency one')
    k = len(gens)
    if phi is None:
        phi = MapToGroupRingOfFreeAbelianization(G, F)
    if len(phi.range().gens()) != 1:
        raise ValueError('Algorithm to compute torsion requires betti number 1')
    i0 = [i for i, g in enumerate(gens) if phi(g) != 1][0]
    gens = gens[i0:] + gens[:i0]
    if phialpha is None:
        phialpha = PhiAlpha(phi, alpha)
    if not wada_conventions:
        d2 = [[fox_derivative_with_involution(R, phialpha, g) for R in rels] for g in gens]
        d2 = block_matrix(d2, nrows=k, ncols=k - 1)
        d1 = [phialpha(g.swapcase()) - 1 for g in gens]
        d1 = block_matrix(d1, nrows=1, ncols=k)
        dsquared = d1 * d2
    else:
        d2 = [[fox_derivative(R, phialpha, g) for g in gens] for R in rels]
        d2 = block_matrix(sum(d2, []), nrows=k - 1, ncols=k)
        d1 = [phialpha(g) - 1 for g in gens]
        d1 = block_matrix(d1, nrows=k, ncols=1)
        dsquared = d2 * d1
    if not matrix_has_small_entries(dsquared, epsilon):
        raise TorsionComputationError('(boundary)^2 != 0')
    T = last_square_submatrix(d2)
    if return_as_poly:
        T = fast_determinant_of_laurent_poly_matrix(T)
    else:
        T = det(T)
    B = first_square_submatrix(d1)
    B = det(B)
    if return_as_poly:
        T = clean_laurent_to_poly(T, epsilon)
        B = clean_laurent_to_poly(B, epsilon)
    else:
        T = clean_laurent(T, epsilon)
        B = clean_laurent(B, epsilon)
    if return_parts:
        return (T, B)
    q, r = T.quo_rem(B)
    ans = clean_laurent_to_poly(q, epsilon)
    if not F.is_exact() and univ_abs(r) > epsilon or (F.is_exact() and r != 0):
        raise TorsionComputationError('Division failed')
    if symmetry_test:
        coeffs = ans.coefficients()
        error = max([univ_abs(a - b) for a, b in zip(coeffs, reversed(coeffs))], default=0)
        if not F.is_exact() and error > epsilon or (F.is_exact() and error != 0):
            raise TorsionComputationError("Torsion polynomial doesn't seem symmetric")
    return ans