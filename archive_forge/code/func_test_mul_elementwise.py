from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
import pytest
import scipy.sparse as sp
import cvxpy.settings as s
from cvxpy.lin_ops.canon_backend import (
def test_mul_elementwise(self, backend):
    """
        define x = Variable((2,)) with
        [x1, x2]

        x is represented as eye(2) in the A matrix, i.e.,

         x1  x2
        [[1  0],
         [0  1]]

         mul_elementwise(x, a) means 'a' is reshaped into a column vector and multiplied by A.
         E.g. for a = (2,3), we obtain

         x1  x2
        [[2  0],
         [0  3]]
        """
    variable_lin_op = linOpHelper((2,), type='variable', data=1)
    view = backend.process_constraint(variable_lin_op, backend.get_empty_view())
    view_A = view.get_tensor_representation(0, 2)
    view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(2, 2)).toarray()
    assert np.all(view_A == np.eye(2))
    lhs = linOpHelper((2,), type='dense_const', data=np.array([2, 3]))
    mul_elementwise_lin_op = linOpHelper(data=lhs)
    out_view = backend.mul_elem(mul_elementwise_lin_op, view)
    A = out_view.get_tensor_representation(0, 2)
    A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(2, 2)).toarray()
    expected = np.array([[2, 0], [0, 3]])
    assert np.all(A == expected)
    assert out_view.get_tensor_representation(0, 2) == view.get_tensor_representation(0, 2)