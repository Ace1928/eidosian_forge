from ..sage_helper import _within_sage, sage_method
from .. import SnapPy
def dim_twisted_homology(group, matrix_of_C_p_rep):
    """
       sage: M = Manifold('K12n813')
       sage: G = M.fundamental_group()
       sage: reps = irreps(3, 7)
       sage: [dim_twisted_homology(G, A) for A in reps]
       [1, 1, 1]
       sage: reps = irreps(3, 5)
       sage: [dim_twisted_homology(G, A) for A in reps]
       [1, 0]
    """
    rho = cyclic_rep(group, matrix_of_C_p_rep)
    C = rho.twisted_chain_complex()
    H = C.homology()
    if matrix_of_C_p_rep != 1:
        assert H[0].rank() == 0
    else:
        assert H[0].rank() == 1
    return H[1].rank()