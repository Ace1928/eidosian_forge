from warnings import warn
import numpy as np
from numpy.linalg import pinv
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import splu
from scipy.optimize import OptimizeResult
def estimate_fun_jac(fun, x, y, p, f0=None):
    """Estimate derivatives of an ODE system rhs with forward differences.

    Returns
    -------
    df_dy : ndarray, shape (n, n, m)
        Derivatives with respect to y. An element (i, j, q) corresponds to
        d f_i(x_q, y_q) / d (y_q)_j.
    df_dp : ndarray with shape (n, k, m) or None
        Derivatives with respect to p. An element (i, j, q) corresponds to
        d f_i(x_q, y_q, p) / d p_j. If `p` is empty, None is returned.
    """
    n, m = y.shape
    if f0 is None:
        f0 = fun(x, y, p)
    dtype = y.dtype
    df_dy = np.empty((n, n, m), dtype=dtype)
    h = EPS ** 0.5 * (1 + np.abs(y))
    for i in range(n):
        y_new = y.copy()
        y_new[i] += h[i]
        hi = y_new[i] - y[i]
        f_new = fun(x, y_new, p)
        df_dy[:, i, :] = (f_new - f0) / hi
    k = p.shape[0]
    if k == 0:
        df_dp = None
    else:
        df_dp = np.empty((n, k, m), dtype=dtype)
        h = EPS ** 0.5 * (1 + np.abs(p))
        for i in range(k):
            p_new = p.copy()
            p_new[i] += h[i]
            hi = p_new[i] - p[i]
            f_new = fun(x, y, p_new)
            df_dp[:, i, :] = (f_new - f0) / hi
    return (df_dy, df_dp)