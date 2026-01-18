from warnings import warn
import numpy as np
from numpy.linalg import pinv
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import splu
from scipy.optimize import OptimizeResult
def construct_global_jac(n, m, k, i_jac, j_jac, h, df_dy, df_dy_middle, df_dp, df_dp_middle, dbc_dya, dbc_dyb, dbc_dp):
    """Construct the Jacobian of the collocation system.

    There are n * m + k functions: m - 1 collocations residuals, each
    containing n components, followed by n + k boundary condition residuals.

    There are n * m + k variables: m vectors of y, each containing n
    components, followed by k values of vector p.

    For example, let m = 4, n = 2 and k = 1, then the Jacobian will have
    the following sparsity structure:

        1 1 2 2 0 0 0 0  5
        1 1 2 2 0 0 0 0  5
        0 0 1 1 2 2 0 0  5
        0 0 1 1 2 2 0 0  5
        0 0 0 0 1 1 2 2  5
        0 0 0 0 1 1 2 2  5

        3 3 0 0 0 0 4 4  6
        3 3 0 0 0 0 4 4  6
        3 3 0 0 0 0 4 4  6

    Zeros denote identically zero values, other values denote different kinds
    of blocks in the matrix (see below). The blank row indicates the separation
    of collocation residuals from boundary conditions. And the blank column
    indicates the separation of y values from p values.

    Refer to [1]_  (p. 306) for the formula of n x n blocks for derivatives
    of collocation residuals with respect to y.

    Parameters
    ----------
    n : int
        Number of equations in the ODE system.
    m : int
        Number of nodes in the mesh.
    k : int
        Number of the unknown parameters.
    i_jac, j_jac : ndarray
        Row and column indices returned by `compute_jac_indices`. They
        represent different blocks in the Jacobian matrix in the following
        order (see the scheme above):

            * 1: m - 1 diagonal n x n blocks for the collocation residuals.
            * 2: m - 1 off-diagonal n x n blocks for the collocation residuals.
            * 3 : (n + k) x n block for the dependency of the boundary
              conditions on ya.
            * 4: (n + k) x n block for the dependency of the boundary
              conditions on yb.
            * 5: (m - 1) * n x k block for the dependency of the collocation
              residuals on p.
            * 6: (n + k) x k block for the dependency of the boundary
              conditions on p.

    df_dy : ndarray, shape (n, n, m)
        Jacobian of f with respect to y computed at the mesh nodes.
    df_dy_middle : ndarray, shape (n, n, m - 1)
        Jacobian of f with respect to y computed at the middle between the
        mesh nodes.
    df_dp : ndarray with shape (n, k, m) or None
        Jacobian of f with respect to p computed at the mesh nodes.
    df_dp_middle : ndarray with shape (n, k, m - 1) or None
        Jacobian of f with respect to p computed at the middle between the
        mesh nodes.
    dbc_dya, dbc_dyb : ndarray, shape (n, n)
        Jacobian of bc with respect to ya and yb.
    dbc_dp : ndarray with shape (n, k) or None
        Jacobian of bc with respect to p.

    Returns
    -------
    J : csc_matrix, shape (n * m + k, n * m + k)
        Jacobian of the collocation system in a sparse form.

    References
    ----------
    .. [1] J. Kierzenka, L. F. Shampine, "A BVP Solver Based on Residual
       Control and the Maltab PSE", ACM Trans. Math. Softw., Vol. 27,
       Number 3, pp. 299-316, 2001.
    """
    df_dy = np.transpose(df_dy, (2, 0, 1))
    df_dy_middle = np.transpose(df_dy_middle, (2, 0, 1))
    h = h[:, np.newaxis, np.newaxis]
    dtype = df_dy.dtype
    dPhi_dy_0 = np.empty((m - 1, n, n), dtype=dtype)
    dPhi_dy_0[:] = -np.identity(n)
    dPhi_dy_0 -= h / 6 * (df_dy[:-1] + 2 * df_dy_middle)
    T = stacked_matmul(df_dy_middle, df_dy[:-1])
    dPhi_dy_0 -= h ** 2 / 12 * T
    dPhi_dy_1 = np.empty((m - 1, n, n), dtype=dtype)
    dPhi_dy_1[:] = np.identity(n)
    dPhi_dy_1 -= h / 6 * (df_dy[1:] + 2 * df_dy_middle)
    T = stacked_matmul(df_dy_middle, df_dy[1:])
    dPhi_dy_1 += h ** 2 / 12 * T
    values = np.hstack((dPhi_dy_0.ravel(), dPhi_dy_1.ravel(), dbc_dya.ravel(), dbc_dyb.ravel()))
    if k > 0:
        df_dp = np.transpose(df_dp, (2, 0, 1))
        df_dp_middle = np.transpose(df_dp_middle, (2, 0, 1))
        T = stacked_matmul(df_dy_middle, df_dp[:-1] - df_dp[1:])
        df_dp_middle += 0.125 * h * T
        dPhi_dp = -h / 6 * (df_dp[:-1] + df_dp[1:] + 4 * df_dp_middle)
        values = np.hstack((values, dPhi_dp.ravel(), dbc_dp.ravel()))
    J = coo_matrix((values, (i_jac, j_jac)))
    return csc_matrix(J)