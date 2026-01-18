import numpy as np
from scipy.linalg import (norm, get_lapack_funcs, solve_triangular,
from ._trustregion import (_minimize_trust_region, BaseQuadraticSubproblem)
def gershgorin_bounds(H):
    """
    Given a square matrix ``H`` compute upper
    and lower bounds for its eigenvalues (Gregoshgorin Bounds).
    Defined ref. [1].

    References
    ----------
    .. [1] Conn, A. R., Gould, N. I., & Toint, P. L.
           Trust region methods. 2000. Siam. pp. 19.
    """
    H_diag = np.diag(H)
    H_diag_abs = np.abs(H_diag)
    H_row_sums = np.sum(np.abs(H), axis=1)
    lb = np.min(H_diag + H_diag_abs - H_row_sums)
    ub = np.max(H_diag - H_diag_abs + H_row_sums)
    return (lb, ub)