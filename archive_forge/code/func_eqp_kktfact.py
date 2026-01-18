from scipy.sparse import (linalg, bmat, csc_matrix)
from math import copysign
import numpy as np
from numpy.linalg import norm
def eqp_kktfact(H, c, A, b):
    """Solve equality-constrained quadratic programming (EQP) problem.

    Solve ``min 1/2 x.T H x + x.t c`` subject to ``A x + b = 0``
    using direct factorization of the KKT system.

    Parameters
    ----------
    H : sparse matrix, shape (n, n)
        Hessian matrix of the EQP problem.
    c : array_like, shape (n,)
        Gradient of the quadratic objective function.
    A : sparse matrix
        Jacobian matrix of the EQP problem.
    b : array_like, shape (m,)
        Right-hand side of the constraint equation.

    Returns
    -------
    x : array_like, shape (n,)
        Solution of the KKT problem.
    lagrange_multipliers : ndarray, shape (m,)
        Lagrange multipliers of the KKT problem.
    """
    n, = np.shape(c)
    m, = np.shape(b)
    kkt_matrix = csc_matrix(bmat([[H, A.T], [A, None]]))
    kkt_vec = np.hstack([-c, -b])
    lu = linalg.splu(kkt_matrix)
    kkt_sol = lu.solve(kkt_vec)
    x = kkt_sol[:n]
    lagrange_multipliers = -kkt_sol[n:n + m]
    return (x, lagrange_multipliers)