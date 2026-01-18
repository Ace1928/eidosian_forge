from ..sage_helper import _within_sage, sage_method
from .. import SnapPy
def semidirect_rep_from_twisted_cocycle(self, cocycle):
    """
        Given a representation rho to GL(R, n) and a rho-twisted
        1-cocycle, construct the representation to GL(R, n + 1)
        corresponding to the semidirect product.

        Note: Since we prefer to stick to left-actions only, unlike [HLK]
        this is the semidirect produce associated to the left action of
        GL(R, n) on V = R^n.  That is, pairs (v, A) with v in V and A in
        GL(R, n) where (v, A) * (w, B) = (v + A*w, A*B)::

           sage: G = Manifold('K12a169').fundamental_group()
           sage: A = matrix(GF(5), [[0, 4], [1, 4]])
           sage: rho = cyclic_rep(G, A)
           sage: cocycle = vector(GF(5), (0, 0, 1, 0))
           sage: rho_til = rho.semidirect_rep_from_twisted_cocycle(cocycle)
           sage: rho_til('abAB')
           [1 0 4]
           [0 1 1]
           [0 0 1]
        """
    gens, rels, rho = (self.generators, self.relators, self)
    n = rho.dim
    assert len(cocycle) == len(gens) * n
    new_mats = []
    for i, g in enumerate(gens):
        v = matrix([cocycle[i * n:(i + 1) * n]]).transpose()
        zeros = matrix(n * [0])
        one = matrix([[1]])
        A = block_matrix([[rho(g), v], [zeros, one]])
        new_mats.append(A)
    target = MatrixSpace(rho.base_ring, n + 1)
    return MatrixRepresentation(gens, rels, target, new_mats)