import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse import issparse, csc_matrix, eye
from scipy.sparse.linalg import splu
from scipy.optimize._numdiff import group_columns
from .common import (validate_max_step, validate_tol, select_initial_step,
from .base import OdeSolver, DenseOutput
def solve_bdf_system(fun, t_new, y_predict, c, psi, LU, solve_lu, scale, tol):
    """Solve the algebraic system resulting from BDF method."""
    d = 0
    y = y_predict.copy()
    dy_norm_old = None
    converged = False
    for k in range(NEWTON_MAXITER):
        f = fun(t_new, y)
        if not np.all(np.isfinite(f)):
            break
        dy = solve_lu(LU, c * f - psi - d)
        dy_norm = norm(dy / scale)
        if dy_norm_old is None:
            rate = None
        else:
            rate = dy_norm / dy_norm_old
        if rate is not None and (rate >= 1 or rate ** (NEWTON_MAXITER - k) / (1 - rate) * dy_norm > tol):
            break
        y += dy
        d += dy
        if dy_norm == 0 or (rate is not None and rate / (1 - rate) * dy_norm < tol):
            converged = True
            break
        dy_norm_old = dy_norm
    return (converged, k + 1, y, d)