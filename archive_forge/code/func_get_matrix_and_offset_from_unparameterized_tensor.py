from __future__ import annotations
import numbers
import os
import numpy as np
import scipy.sparse as sp
import cvxpy.cvxcore.python.cvxcore as cvxcore
import cvxpy.settings as s
from cvxpy.lin_ops import lin_op as lo
from cvxpy.lin_ops.canon_backend import CanonBackend
def get_matrix_and_offset_from_unparameterized_tensor(problem_data_tensor, var_length):
    """Converts unparameterized tensor to matrix offset representation

    problem_data_tensor _must_ have been obtained from calling
    get_problem_matrix on a problem with 0 parameters.

    Parameters
    ----------
        problem_data_tensor: tensor returned from get_problem_matrix,
            representing an affine map
        var_length: the number of variables

    Returns
    -------
        A tuple (A, b), where A is a matrix with `var_length` columns
        and b is a flattened NumPy array representing the constant offset.
    """
    assert problem_data_tensor.shape[1] == 1
    return get_matrix_and_offset_from_tensor(problem_data_tensor, None, var_length)