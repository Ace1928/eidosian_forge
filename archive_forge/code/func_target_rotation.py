import numpy as np
import scipy as sp
def target_rotation(A, H, full_rank=False):
    """
    Analytically performs orthogonal rotations towards a target matrix,
    i.e., we minimize:

    .. math::
        \\phi(L) =\\frac{1}{2}\\|AT-H\\|^2.

    where :math:`T` is an orthogonal matrix. This problem is also known as
    an orthogonal Procrustes problem.

    Under the assumption that :math:`A^*H` has full rank, the analytical
    solution :math:`T` is given by:

    .. math::
        T = (A^*HH^*A)^{-\\frac{1}{2}}A^*H,

    see Green (1952). In other cases the solution is given by :math:`T = UV`,
    where :math:`U` and :math:`V` result from the singular value decomposition
    of :math:`A^*H`:

    .. math::
        A^*H = U\\Sigma V,

    see Schonemann (1966).

    Parameters
    ----------
    A : numpy matrix (default None)
        non rotated factors
    H : numpy matrix
        target matrix
    full_rank : bool (default FAlse)
        if set to true full rank is assumed

    Returns
    -------
    The matrix :math:`T`.

    References
    ----------
    [1] Green (1952, Psychometrika) - The orthogonal approximation of an
    oblique structure in factor analysis

    [2] Schonemann (1966) - A generalized solution of the orthogonal
    procrustes problem

    [3] Gower, Dijksterhuis (2004) - Procrustes problems
    """
    ATH = A.T.dot(H)
    if full_rank or np.linalg.matrix_rank(ATH) == A.shape[1]:
        T = sp.linalg.fractional_matrix_power(ATH.dot(ATH.T), -1 / 2).dot(ATH)
    else:
        U, D, V = np.linalg.svd(ATH, full_matrices=False)
        T = U.dot(V)
    return T