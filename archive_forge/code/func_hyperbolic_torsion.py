import string
from ..sage_helper import _within_sage, sage_method
@sage_method
def hyperbolic_torsion(manifold, bits_prec=100, all_lifts=False, wada_conventions=False, phi=None):
    """
    Computes the hyperbolic torsion polynomial as defined in
    `[DFJ] <http://arxiv.org/abs/1108.3045>`_::

        sage: M = Manifold('K11n42')
        sage: M.alexander_polynomial()
        1
        sage: tau = M.hyperbolic_torsion(bits_prec=200)
        sage: tau.degree()
        6
    """
    if manifold.homology().betti_number() != 1:
        raise ValueError('Algorithm needs H^1(M; Z) = Z to be able to compute torsion')
    H = manifold.fundamental_group()
    if H.num_generators() != H.num_relators() + 1:
        raise ValueError('Algorithm to compute torsion requires a group presentation with deficiency one')
    G = alpha = polished_holonomy(manifold, bits_prec=bits_prec, lift_to_SL2=True)
    if not all_lifts:
        return compute_torsion(G, bits_prec, alpha, phi, wada_conventions=wada_conventions)
    else:
        return [compute_torsion(G, bits_prec, beta, phi, wada_conventions=wada_conventions) for beta in alpha.all_lifts_to_SL2C()]