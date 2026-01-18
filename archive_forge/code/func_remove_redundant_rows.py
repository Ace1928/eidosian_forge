from typing import Dict, List, Union
import numpy as np
import scipy
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import cvxpy.interface as intf
import cvxpy.settings as s
from cvxpy.constraints import PSD, SOC, NonNeg, Zero
from cvxpy.reductions.solvers.compr_matrix import compress_matrix
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.reductions.solvers.kktsolver import setup_ldl_factor
@staticmethod
def remove_redundant_rows(data):
    """Check if A has redundant rows. If it does, remove redundant constraints
        from A, and apply a presolve procedure for G.

        Parameters
        ----------
        data : dict
            All the problem data.

        Returns
        -------
        str
            A status indicating if infeasibility was detected.
        """
    dims = data[s.DIMS]
    A = data[s.A]
    G = data[s.G]
    b = data[s.B]
    h = data[s.H]
    if A is None:
        return s.OPTIMAL
    TOL = 1e-10
    gram = A @ A.T
    if gram.shape[0] == 1:
        gram = gram.toarray().item()
        if gram > 0:
            return s.OPTIMAL
        elif not b.item() == 0.0:
            return s.INFEASIBLE
        else:
            data[s.A] = None
            data[s.B] = None
            return s.OPTIMAL
    if hasattr(np.random, 'default_rng'):
        g = np.random.default_rng(123)
    else:
        g = np.random.RandomState(123)
    n = gram.shape[0]
    rand_v0 = g.normal(loc=0.0, scale=1.0, size=n)
    eig = eigsh(gram, k=1, which='SM', v0=rand_v0, return_eigenvectors=False)
    if eig > TOL:
        return s.OPTIMAL
    Q, R, P = scipy.linalg.qr(A.todense(), pivoting=True)
    rows_to_keep = []
    for i in range(R.shape[0]):
        if np.linalg.norm(R[i, :]) > TOL:
            rows_to_keep.append(i)
    R = R[rows_to_keep, :]
    Q = Q[:, rows_to_keep]
    Pinv = np.zeros(P.size, dtype='int')
    for i in range(P.size):
        Pinv[P[i]] = i
    R = R[:, Pinv]
    A = R
    b_old = b
    b = Q.T.dot(b)
    if not np.allclose(b_old, Q.dot(b)):
        return s.INFEASIBLE
    dims[s.EQ_DIM] = int(b.shape[0])
    data['Q'] = intf.dense2cvxopt(Q)
    if G is not None:
        G = G.tocsr()
        G_leq = G[:dims[s.LEQ_DIM], :]
        h_leq = h[:dims[s.LEQ_DIM]].ravel()
        G_other = G[dims[s.LEQ_DIM]:, :]
        h_other = h[dims[s.LEQ_DIM]:].ravel()
        G_leq, h_leq, P_leq = compress_matrix(G_leq, h_leq)
        dims[s.LEQ_DIM] = int(h_leq.shape[0])
        data['P_leq'] = intf.sparse2cvxopt(P_leq)
        G = sp.vstack([G_leq, G_other])
        h = np.hstack([h_leq, h_other])
    data[s.A] = A
    data[s.G] = G
    data[s.B] = b
    data[s.H] = h
    return s.OPTIMAL