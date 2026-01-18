from __future__ import annotations
import numbers
import os
import numpy as np
import scipy.sparse as sp
import cvxpy.cvxcore.python.cvxcore as cvxcore
import cvxpy.settings as s
from cvxpy.lin_ops import lin_op as lo
from cvxpy.lin_ops.canon_backend import CanonBackend
def get_matrix_from_tensor(problem_data_tensor, param_vec, var_length, nonzero_rows=None, with_offset=True, problem_data_index=None):
    """Applies problem_data_tensor to param_vec to obtain matrix and (optionally)
    the offset.

    This function applies problem_data_tensor to param_vec to obtain
    a matrix representation of the corresponding affine map.

    Parameters
    ----------
        problem_data_tensor: tensor returned from get_problem_matrix,
            representing a parameterized affine map
        param_vec: flattened parameter vector
        var_length: the number of variables
        nonzero_rows: (optional) rows in the part of problem_data_tensor
            corresponding to A that have nonzeros in them (i.e., rows that
            are affected by parameters); if not None, then the corresponding
            entries in A will have explicit zeros.
        with_offset: (optional) return offset. Defaults to True.
        problem_data_index: (optional) a tuple (indices, indptr, shape) for
            construction of the CSC matrix holding the problem data and offset

    Returns
    -------
        A tuple (A, b), where A is a matrix with `var_length` columns
        and b is a flattened NumPy array representing the constant offset.
        If with_offset=False, returned b is None.
    """
    if param_vec is None:
        flat_problem_data = problem_data_tensor
        if problem_data_index is not None:
            flat_problem_data = flat_problem_data.toarray().flatten()
    elif problem_data_index is not None:
        flat_problem_data = problem_data_tensor @ param_vec
    else:
        param_vec = sp.csc_matrix(param_vec[:, None])
        flat_problem_data = problem_data_tensor @ param_vec
    if problem_data_index is not None:
        indices, indptr, shape = problem_data_index
        M = sp.csc_matrix((flat_problem_data, indices, indptr), shape=shape)
    else:
        n_cols = var_length
        if with_offset:
            n_cols += 1
        M = flat_problem_data.reshape((-1, n_cols), order='F').tocsc()
    if with_offset:
        A = M[:, :-1].tocsc()
        b = np.squeeze(M[:, -1].toarray().flatten())
    else:
        A = M.tocsc()
        b = None
    if nonzero_rows is not None and nonzero_rows.size > 0:
        A_nrows, _ = A.shape
        A_rows, A_cols = nonzero_csc_matrix(A)
        A_vals = np.append(A.data, np.zeros(nonzero_rows.size))
        A_rows = np.append(A_rows, nonzero_rows % A_nrows)
        A_cols = np.append(A_cols, nonzero_rows // A_nrows)
        A = sp.csc_matrix((A_vals, (A_rows, A_cols)), shape=A.shape)
    return (A, b)