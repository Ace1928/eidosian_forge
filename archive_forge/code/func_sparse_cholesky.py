import cvxpy.utilities.cpp.sparsecholesky as spchol  # noqa: I001
import cvxpy.settings as settings
import numpy as np
import scipy.linalg as la
import scipy.sparse as spar
import scipy.sparse.linalg as sparla
from scipy.sparse import csc_matrix
def sparse_cholesky(A, sym_tol=settings.CHOL_SYM_TOL, assume_posdef=False):
    """
    The input matrix A must be real and symmetric. If A is positive definite then
    Eigen will be used to compute its sparse Cholesky decomposition with AMD-ordering.
    If A is negative definite, then the analogous operation will be applied to -A.

    If Cholesky succeeds, then we return a lower-triangular matrix L in
    CSR-format and a permutation vector p so (L[p, :]) @ (L[p, :]).T == A
    within numerical precision.

    We raise a ValueError if Eigen's Cholesky fails or if we certify indefiniteness
    before calling Eigen. While checking for indefiniteness, we also check that
     ||A - A'||_Fro / sqrt(n) <= sym_tol, where n is the order of the matrix.
    """
    if not isinstance(A, spar.spmatrix):
        raise ValueError(SparseCholeskyMessages.NOT_SPARSE)
    if np.iscomplexobj(A):
        raise ValueError(SparseCholeskyMessages.NOT_REAL)
    if not assume_posdef:
        symdiff = A - A.T
        sz = symdiff.data.size
        if sz > 0 and la.norm(symdiff.data) > sym_tol * sz ** 0.5:
            raise ValueError(SparseCholeskyMessages.ASYMMETRIC)
        d = A.diagonal()
        maybe_posdef = np.all(d > 0)
        maybe_negdef = np.all(d < 0)
        if not (maybe_posdef or maybe_negdef):
            raise ValueError(SparseCholeskyMessages.INDEFINITE)
        if maybe_negdef:
            _, L, p = sparse_cholesky(-A, sym_tol, assume_posdef=True)
            return (-1.0, L, p)
    A_coo = spar.coo_matrix(A)
    n = A.shape[0]
    inrows = spchol.IntVector(A_coo.row)
    incols = spchol.IntVector(A_coo.col)
    invals = spchol.DoubleVector(A_coo.data)
    outpivs = spchol.IntVector()
    outrows = spchol.IntVector()
    outcols = spchol.IntVector()
    outvals = spchol.DoubleVector()
    try:
        spchol.sparse_chol_from_vecs(n, inrows, incols, invals, outpivs, outrows, outcols, outvals)
    except RuntimeError as e:
        if e.args[0] == SparseCholeskyMessages.EIGEN_FAIL:
            raise ValueError(e.args)
        else:
            raise RuntimeError(e.args)
    outvals = np.array(outvals)
    outrows = np.array(outrows)
    outcols = np.array(outcols)
    outpivs = np.array(outpivs)
    L = spar.csr_matrix((outvals, (outrows, outcols)), shape=(n, n))
    return (1.0, L, outpivs)