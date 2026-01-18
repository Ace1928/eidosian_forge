import numpy as np
from scipy.linalg import solve_sylvester
import pandas as pd
from statsmodels.compat.pandas import Appender
from statsmodels.tools.data import _is_using_pandas
from scipy.linalg.blas import find_best_blas_type
from . import (_initialization, _representation, _kalman_filter,
def companion_matrix(polynomial):
    """
    Create a companion matrix

    Parameters
    ----------
    polynomial : array_like or list
        If an iterable, interpreted as the coefficients of the polynomial from
        which to form the companion matrix. Polynomial coefficients are in
        order of increasing degree, and may be either scalars (as in an AR(p)
        model) or coefficient matrices (as in a VAR(p) model). If an integer,
        it is interpreted as the size of a companion matrix of a scalar
        polynomial, where the polynomial coefficients are initialized to zeros.
        If a matrix polynomial is passed, :math:`C_0` may be set to the scalar
        value 1 to indicate an identity matrix (doing so will improve the speed
        of the companion matrix creation).

    Returns
    -------
    companion_matrix : ndarray

    Notes
    -----
    Given coefficients of a lag polynomial of the form:

    .. math::

        c(L) = c_0 + c_1 L + \\dots + c_p L^p

    returns a matrix of the form

    .. math::
        \\begin{bmatrix}
            \\phi_1 & 1      & 0 & \\cdots & 0 \\\\
            \\phi_2 & 0      & 1 &        & 0 \\\\
            \\vdots &        &   & \\ddots & 0 \\\\
                   &        &   &        & 1 \\\\
            \\phi_n & 0      & 0 & \\cdots & 0 \\\\
        \\end{bmatrix}

    where some or all of the :math:`\\phi_i` may be non-zero (if `polynomial` is
    None, then all are equal to zero).

    If the coefficients provided are scalars :math:`(c_0, c_1, \\dots, c_p)`,
    then the companion matrix is an :math:`n \\times n` matrix formed with the
    elements in the first column defined as
    :math:`\\phi_i = -\\frac{c_i}{c_0}, i \\in 1, \\dots, p`.

    If the coefficients provided are matrices :math:`(C_0, C_1, \\dots, C_p)`,
    each of shape :math:`(m, m)`, then the companion matrix is an
    :math:`nm \\times nm` matrix formed with the elements in the first column
    defined as :math:`\\phi_i = -C_0^{-1} C_i', i \\in 1, \\dots, p`.

    It is important to understand the expected signs of the coefficients. A
    typical AR(p) model is written as:

    .. math::
        y_t = a_1 y_{t-1} + \\dots + a_p y_{t-p} + \\varepsilon_t

    This can be rewritten as:

    .. math::
        (1 - a_1 L - \\dots - a_p L^p )y_t = \\varepsilon_t \\\\
        (1 + c_1 L + \\dots + c_p L^p )y_t = \\varepsilon_t \\\\
        c(L) y_t = \\varepsilon_t

    The coefficients from this form are defined to be :math:`c_i = - a_i`, and
    it is the :math:`c_i` coefficients that this function expects to be
    provided.
    """
    identity_matrix = False
    if isinstance(polynomial, (int, np.integer)):
        n = int(polynomial)
        m = 1
        polynomial = None
    else:
        n = len(polynomial) - 1
        if n < 1:
            raise ValueError('Companion matrix polynomials must include at least two terms.')
        if isinstance(polynomial, (list, tuple)):
            try:
                m = len(polynomial[1])
            except TypeError:
                m = 1
            if m == 1:
                polynomial = np.asanyarray(polynomial)
            elif polynomial[0] == 1:
                polynomial[0] = np.eye(m)
                identity_matrix = True
        else:
            m = 1
            polynomial = np.asanyarray(polynomial)
    matrix = np.zeros((n * m, n * m), dtype=np.asanyarray(polynomial).dtype)
    idx = np.diag_indices((n - 1) * m)
    idx = (idx[0], idx[1] + m)
    matrix[idx] = 1
    if polynomial is not None and n > 0:
        if m == 1:
            matrix[:, 0] = -polynomial[1:] / polynomial[0]
        elif identity_matrix:
            for i in range(n):
                matrix[i * m:(i + 1) * m, :m] = -polynomial[i + 1].T
        else:
            inv = np.linalg.inv(polynomial[0])
            for i in range(n):
                matrix[i * m:(i + 1) * m, :m] = -np.dot(inv, polynomial[i + 1]).T
    return matrix