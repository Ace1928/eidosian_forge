from __future__ import annotations
import numbers
import os
import numpy as np
import scipy.sparse as sp
import cvxpy.cvxcore.python.cvxcore as cvxcore
import cvxpy.settings as s
from cvxpy.lin_ops import lin_op as lo
from cvxpy.lin_ops.canon_backend import CanonBackend
def reduce_problem_data_tensor(A, var_length, quad_form: bool=False):
    """Reduce a problem data tensor, for efficient construction of the problem data

    If quad_form=False, the problem data tensor A is a matrix of shape (m, p), where p is the
    length of the parameter vector. The product A@param_vec gives the
    entries of the problem data matrix for a solver;
    the solver's problem data matrix has dimensions(n_constr, n_var + 1),
    and n_constr*(n_var + 1) = m. In other words, each row in A corresponds
    to an entry in the solver's data matrix.

    If quad_form=True, the problem data tensor A is a matrix of shape (m, p), where p is the
    length of the parameter vector. The product A@param_vec gives the
    entries of the quadratic form matrix P for a QP solver;
    the solver's quadratic matrix P has dimensions(n_var, n_var),
    and n_var*n_var = m. In other words, each row in A corresponds
    to an entry in the solver's quadratic form matrix P.

    This function removes the rows in A that are identically zero, since these
    rows correspond to zeros in the problem data. It also returns the indices
    and indptr to construct the problem data matrix from the reduced
    representation of A, and the shape of the problem data matrix.

    Let reduced_A be the sparse matrix returned by this function. Then the
    problem data can be computed using

       data : = reduced_A @ param_vec

    and the problem data matrix can be constructed with

        problem_data : = sp.csc_matrix(
            (data, indices, indptr), shape = shape)

    Parameters
    ----------
        A : A sparse matrix, the problem data tensor; must not have a 0 in its
            shape
        var_length: number of variables in the problem
        quad_form: (optional) if True, consider quadratic form matrix P

    Returns
    -------
        reduced_A: A CSR sparse matrix with redundant rows removed
        indices: CSC indices for the problem data matrix
        indptr: CSC indptr for the problem data matrix
        shape: the shape of the problem data matrix
    """
    A.eliminate_zeros()
    A_coo = A.tocoo()
    unique_old_row, reduced_row = np.unique(A_coo.row, return_inverse=True)
    reduced_A_shape = (unique_old_row.size, A_coo.shape[1])
    reduced_A = sp.coo_matrix((A_coo.data, (reduced_row, A_coo.col)), shape=reduced_A_shape)
    reduced_A = reduced_A.tocsr()
    nonzero_rows = unique_old_row
    n_cols = var_length
    if not quad_form:
        n_cols += 1
    n_constr = A.shape[0] // n_cols
    shape = (n_constr, n_cols)
    indices = nonzero_rows % n_constr
    cols = nonzero_rows // n_constr
    indptr = np.zeros(n_cols + 1, dtype=np.int32)
    positions, counts = np.unique(cols, return_counts=True)
    indptr[positions + 1] = counts
    indptr = np.cumsum(indptr)
    return (reduced_A, indices, indptr, shape)