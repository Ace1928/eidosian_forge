from warnings import warn
import numpy as np
from numpy.linalg import pinv
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import splu
from scipy.optimize import OptimizeResult
def estimate_rms_residuals(fun, sol, x, h, p, r_middle, f_middle):
    """Estimate rms values of collocation residuals using Lobatto quadrature.

    The residuals are defined as the difference between the derivatives of
    our solution and rhs of the ODE system. We use relative residuals, i.e.,
    normalized by 1 + np.abs(f). RMS values are computed as sqrt from the
    normalized integrals of the squared relative residuals over each interval.
    Integrals are estimated using 5-point Lobatto quadrature [1]_, we use the
    fact that residuals at the mesh nodes are identically zero.

    In [2] they don't normalize integrals by interval lengths, which gives
    a higher rate of convergence of the residuals by the factor of h**0.5.
    I chose to do such normalization for an ease of interpretation of return
    values as RMS estimates.

    Returns
    -------
    rms_res : ndarray, shape (m - 1,)
        Estimated rms values of the relative residuals over each interval.

    References
    ----------
    .. [1] http://mathworld.wolfram.com/LobattoQuadrature.html
    .. [2] J. Kierzenka, L. F. Shampine, "A BVP Solver Based on Residual
       Control and the Maltab PSE", ACM Trans. Math. Softw., Vol. 27,
       Number 3, pp. 299-316, 2001.
    """
    x_middle = x[:-1] + 0.5 * h
    s = 0.5 * h * (3 / 7) ** 0.5
    x1 = x_middle + s
    x2 = x_middle - s
    y1 = sol(x1)
    y2 = sol(x2)
    y1_prime = sol(x1, 1)
    y2_prime = sol(x2, 1)
    f1 = fun(x1, y1, p)
    f2 = fun(x2, y2, p)
    r1 = y1_prime - f1
    r2 = y2_prime - f2
    r_middle /= 1 + np.abs(f_middle)
    r1 /= 1 + np.abs(f1)
    r2 /= 1 + np.abs(f2)
    r1 = np.sum(np.real(r1 * np.conj(r1)), axis=0)
    r2 = np.sum(np.real(r2 * np.conj(r2)), axis=0)
    r_middle = np.sum(np.real(r_middle * np.conj(r_middle)), axis=0)
    return (0.5 * (32 / 45 * r_middle + 49 / 90 * (r1 + r2))) ** 0.5