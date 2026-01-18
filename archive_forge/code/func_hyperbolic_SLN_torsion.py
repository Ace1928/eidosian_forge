import string
from ..sage_helper import _within_sage, sage_method
@sage_method
def hyperbolic_SLN_torsion(manifold, N, bits_prec=100):
    """
    Compute the torsion polynomial of the holonomy representation lifted
    to SL(2, C) and then followed by the irreducible representation
    from SL(2, C) -> SL(N, C)::

        sage: M = Manifold('m016')
        sage: [M.hyperbolic_SLN_torsion(N).degree() for N in [2, 3, 4]]
        [18, 27, 36]
    """
    if manifold.homology().betti_number() != 1:
        raise ValueError('Algorithm needs H^1(M; Z) = Z to be able to compute torsion')
    H = manifold.fundamental_group()
    if H.num_generators() != H.num_relators() + 1:
        raise ValueError('Algorithm to compute torsion requires a group presentation with deficiency one')
    G = alpha = polished_holonomy(manifold, bits_prec)
    phi = MapToGroupRingOfFreeAbelianization(G, alpha('a').base_ring())
    phialpha = PhiAlphaN(phi, alpha, N)
    if not test_rep(G, phialpha) < ZZ(2) ** (bits_prec // 2):
        raise RuntimeError('Invalid representation')
    return compute_torsion(G, bits_prec, phialpha=phialpha, symmetry_test=False)