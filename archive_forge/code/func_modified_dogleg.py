from scipy.sparse import (linalg, bmat, csc_matrix)
from math import copysign
import numpy as np
from numpy.linalg import norm
def modified_dogleg(A, Y, b, trust_radius, lb, ub):
    """Approximately  minimize ``1/2*|| A x + b ||^2`` inside trust-region.

    Approximately solve the problem of minimizing ``1/2*|| A x + b ||^2``
    subject to ``||x|| < Delta`` and ``lb <= x <= ub`` using a modification
    of the classical dogleg approach.

    Parameters
    ----------
    A : LinearOperator (or sparse matrix or ndarray), shape (m, n)
        Matrix ``A`` in the minimization problem. It should have
        dimension ``(m, n)`` such that ``m < n``.
    Y : LinearOperator (or sparse matrix or ndarray), shape (n, m)
        LinearOperator that apply the projection matrix
        ``Q = A.T inv(A A.T)`` to the vector. The obtained vector
        ``y = Q x`` being the minimum norm solution of ``A y = x``.
    b : array_like, shape (m,)
        Vector ``b``in the minimization problem.
    trust_radius: float
        Trust radius to be considered. Delimits a sphere boundary
        to the problem.
    lb : array_like, shape (n,)
        Lower bounds to each one of the components of ``x``.
        It is expected that ``lb <= 0``, otherwise the algorithm
        may fail. If ``lb[i] = -Inf``, the lower
        bound for the ith component is just ignored.
    ub : array_like, shape (n, )
        Upper bounds to each one of the components of ``x``.
        It is expected that ``ub >= 0``, otherwise the algorithm
        may fail. If ``ub[i] = Inf``, the upper bound for the ith
        component is just ignored.

    Returns
    -------
    x : array_like, shape (n,)
        Solution to the problem.

    Notes
    -----
    Based on implementations described in pp. 885-886 from [1]_.

    References
    ----------
    .. [1] Byrd, Richard H., Mary E. Hribar, and Jorge Nocedal.
           "An interior point algorithm for large-scale nonlinear
           programming." SIAM Journal on Optimization 9.4 (1999): 877-900.
    """
    newton_point = -Y.dot(b)
    if inside_box_boundaries(newton_point, lb, ub) and norm(newton_point) <= trust_radius:
        x = newton_point
        return x
    g = A.T.dot(b)
    A_g = A.dot(g)
    cauchy_point = -np.dot(g, g) / np.dot(A_g, A_g) * g
    origin_point = np.zeros_like(cauchy_point)
    z = cauchy_point
    p = newton_point - cauchy_point
    _, alpha, intersect = box_sphere_intersections(z, p, lb, ub, trust_radius)
    if intersect:
        x1 = z + alpha * p
    else:
        z = origin_point
        p = cauchy_point
        _, alpha, _ = box_sphere_intersections(z, p, lb, ub, trust_radius)
        x1 = z + alpha * p
    z = origin_point
    p = newton_point
    _, alpha, _ = box_sphere_intersections(z, p, lb, ub, trust_radius)
    x2 = z + alpha * p
    if norm(A.dot(x1) + b) < norm(A.dot(x2) + b):
        return x1
    else:
        return x2