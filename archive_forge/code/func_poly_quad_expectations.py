import math
import cmath
import numpy as np
from scipy.special import factorial as fac
import pennylane as qml
from pennylane.ops import Identity
from pennylane import Device
from .._version import __version__
def poly_quad_expectations(cov, mu, wires, device_wires, params, hbar=2.0):
    """Calculates the expectation and variance for an arbitrary
    polynomial of quadrature operators.

    Args:
        cov (array): covariance matrix
        mu (array): vector of means
        wires (Wires): wires to calculate the expectation for
        device_wires (Wires): corresponding wires on the device
        params (array): a :math:`(2N+1)\\times (2N+1)` array containing the linear
            and quadratic coefficients of the quadrature operators
            :math:`(\\I, \\x_0, \\p_0, \\x_1, \\p_1,\\dots)`
        hbar (float): (default 2) the value of :math:`\\hbar` in the commutation
            relation :math:`[\\x,\\p]=i\\hbar`

    Returns:
        tuple: the mean and variance of the quadrature-polynomial observable
    """
    Q = params[0]
    op = qml.PolyXP(Q, wires=wires)
    Q = op.heisenberg_obs(device_wires)
    if Q.ndim == 1:
        d = np.r_[Q[1::2], Q[2::2]]
        return (d.T @ mu + Q[0], d.T @ cov @ d)
    M = np.vstack((Q[0:1, :], Q[1::2, :], Q[2::2, :]))
    M = np.hstack((M[:, 0:1], M[:, 1::2], M[:, 2::2]))
    d1 = M[1:, 0]
    d2 = M[0, 1:]
    A = M[1:, 1:]
    d = d1 + d2
    k = M[0, 0]
    d2 = 2 * A @ mu + d
    k2 = mu.T @ A @ mu + mu.T @ d + k
    ex = np.trace(A @ cov) + k2
    var = 2 * np.trace(A @ cov @ A @ cov) + d2.T @ cov @ d2
    modes = np.arange(2 * len(device_wires)).reshape(2, -1).T
    groenewald_correction = np.sum([np.linalg.det(hbar * A[:, m][n]) for m in modes for n in modes])
    var -= groenewald_correction
    return (ex, var)