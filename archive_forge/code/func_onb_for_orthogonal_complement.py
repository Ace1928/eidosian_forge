import cvxpy.utilities.cpp.sparsecholesky as spchol  # noqa: I001
import cvxpy.settings as settings
import numpy as np
import scipy.linalg as la
import scipy.sparse as spar
import scipy.sparse.linalg as sparla
from scipy.sparse import csc_matrix
def onb_for_orthogonal_complement(V):
    """
    Let U = the orthogonal complement of range(V).

    This function returns an array Q whose columns are
    an orthonormal basis for U. It requires that dim(U) > 0.
    """
    n = V.shape[0]
    Q1 = orth(V)
    rank = Q1.shape[1]
    assert n > rank
    if np.iscomplexobj(V):
        P = np.eye(n) - Q1 @ Q1.conj().T
    else:
        P = np.eye(n) - Q1 @ Q1.T
    Q2 = orth(P)
    return Q2