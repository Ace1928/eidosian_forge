from ..sage_helper import _within_sage, sage_method
from .. import SnapPy
def cyclic_rep(group, matrix_of_rep):
    """
    For a group G whose free abelianization is Z, returns the
    representation of G factoring through Z where 1 in Z in turn goes
    to the given matrix_of_rep.
    """
    A = matrix_of_rep
    epsilon = MapToFreeAbelianization(group)
    assert epsilon.range().rank() == 1
    gens = group.generators()
    rels = group.relators()
    mats = [A ** epsilon(g)[0] for g in gens]
    rho = MatrixRepresentation(gens, rels, A.parent(), mats)
    rho.epsilon = epsilon
    rho.A = A
    return rho