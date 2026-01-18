from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
import pytest
import scipy.sparse as sp
import cvxpy.settings as s
from cvxpy.lin_ops.canon_backend import (
def test_diag_vec_with_offset(self, backend):
    """
        define x = Variable((2,)) with
        [x1, x2]

        x is represented as eye(2) in the A matrix, i.e.,

         x1  x2
        [[1  0],
         [0  1]]

        diag_vec(x, k) means we introduce zero rows as if the vector was the k-diagonal
        of an n+|k| x n+|k| matrix, with n the length of x.

        Thus, for k=1 and using the same columns as before, want to represent
        [[0  x1 0],
        [ 0  0  x2],
        [[0  0  0]]
        i.e., unrolled in column-major order:

         x1  x2
        [[0  0],
        [0  0],
        [0  0],
        [1  0],
        [0  0],
        [0  0],
        [0  0],
        [0  1],
        [0  0]]
        """
    variable_lin_op = linOpHelper((2,), type='variable', data=1)
    view = backend.process_constraint(variable_lin_op, backend.get_empty_view())
    view_A = view.get_tensor_representation(0, 2)
    view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(2, 2)).toarray()
    assert np.all(view_A == np.eye(2))
    k = 1
    diag_vec_lin_op = linOpHelper(shape=(3, 3), data=k)
    out_view = backend.diag_vec(diag_vec_lin_op, view)
    A = out_view.get_tensor_representation(0, 9)
    A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(9, 2)).toarray()
    expected = np.array([[0, 0], [0, 0], [0, 0], [1, 0], [0, 0], [0, 0], [0, 0], [0, 1], [0, 0]])
    assert np.all(A == expected)
    assert out_view.get_tensor_representation(0, 9) == view.get_tensor_representation(0, 9)