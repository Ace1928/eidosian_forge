import cvxpy.utilities.cpp.sparsecholesky as spchol  # noqa: I001
import cvxpy.settings as settings
import numpy as np
import scipy.linalg as la
import scipy.sparse as spar
import scipy.sparse.linalg as sparla
from scipy.sparse import csc_matrix
def is_psd_within_tol(A, tol):
    """
    Return True if we can certify that A is PSD (up to tolerance "tol").

    First we check if A is PSD according to the Gershgorin Circle Theorem.

    If Gershgorin is inconclusive, then we use an iterative method (from ARPACK,
    as called through SciPy) to estimate extremal eigenvalues of certain shifted
    versions of A. The shifts are chosen so that the signs of those eigenvalues
    tell us the signs of the eigenvalues of A.

    If there are numerical issues then it's possible that this function returns
    False even when A is PSD. If you know that you're in that situation, then
    you should replace A by

        A = cvxpy.atoms.affine.wraps.psd_wrap(A).

    Parameters
    ----------
    A : Union[np.ndarray, spar.spmatrix]
        Symmetric (or Hermitian) NumPy ndarray or SciPy sparse matrix.

    tol : float
        Nonnegative. Something very small, like 1e-10.
    """
    if gershgorin_psd_check(A, tol):
        return True
    if is_diagonal(A):
        if isinstance(A, csc_matrix):
            return np.all(A.data >= -tol)
        else:
            min_diag_entry = np.min(np.diag(A))
            return min_diag_entry >= -tol

    def SA_eigsh(sigma):
        if hasattr(np.random, 'default_rng'):
            g = np.random.default_rng(123)
        else:
            g = np.random.RandomState(123)
        n = A.shape[0]
        v0 = g.normal(loc=0.0, scale=1.0, size=n)
        return sparla.eigsh(A, k=1, sigma=sigma, which='SA', v0=v0, return_eigenvectors=False)
    try:
        ev = SA_eigsh(-tol)
    except sparla.ArpackNoConvergence as e:
        message = "\n        CVXPY note: This failure was encountered while trying to certify\n        that a matrix is positive semi-definite (see [1] for a definition).\n        In rare cases, this method fails for numerical reasons even when the matrix is\n        positive semi-definite. If you know that you're in that situation, you can\n        replace the matrix A by cvxpy.psd_wrap(A).\n\n        [1] https://en.wikipedia.org/wiki/Definite_matrix\n        "
        error_with_note = f'{str(e)}\n\n{message}'
        raise sparla.ArpackNoConvergence(error_with_note, e.eigenvalues, e.eigenvectors)
    if np.isnan(ev).any():
        temp = tol - np.finfo(A.dtype).eps
        ev = SA_eigsh(-temp)
    return np.all(ev >= -tol)