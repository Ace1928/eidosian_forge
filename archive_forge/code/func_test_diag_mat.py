from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
import pytest
import scipy.sparse as sp
import cvxpy.settings as s
from cvxpy.lin_ops.canon_backend import (
def test_diag_mat(self, backend):
    """
        define x = Variable((2,2)) with
        [[x11, x12],
         [x21, x22]]

        x is represented as eye(4) in the A matrix (in column-major order), i.e.,

         x11 x21 x12 x22
        [[1   0   0   0],
         [0   1   0   0],
         [0   0   1   0],
         [0   0   0   1]]

        diag_mat(x) means we select only the diagonal, i.e., x11 and x22.

        which, when using the same columns as before, now maps to

         x11 x21 x12 x22
        [[1   0   0   0],
         [0   0   0   1]]

        -> It reduces to selecting a subset of the rows of A.
        """
    variable_lin_op = linOpHelper((2, 2), type='variable', data=1)
    view = backend.process_constraint(variable_lin_op, backend.get_empty_view())
    view_A = view.get_tensor_representation(0, 4)
    view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(4, 4)).toarray()
    assert np.all(view_A == np.eye(4))
    diag_mat_lin_op = linOpHelper(shape=(2, 2), data=0)
    out_view = backend.diag_mat(diag_mat_lin_op, view)
    A = out_view.get_tensor_representation(0, 2)
    A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(2, 4)).toarray()
    expected = np.array([[1, 0, 0, 0], [0, 0, 0, 1]])
    assert np.all(A == expected)
    assert out_view.get_tensor_representation(0, 4) == view.get_tensor_representation(0, 4)