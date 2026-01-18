from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable
import numpy as np
import scipy.sparse as sp
from scipy.signal import convolve
from cvxpy.lin_ops import LinOp
from cvxpy.settings import (
def get_constant_data(self, lin_op: LinOp, view: TensorView, column: bool) -> tuple[np.ndarray | sp.spmatrix, bool]:
    """
        Extract the constant data from a LinOp node. In most cases, lin_op will be of
        type "*_const" or "param", but can handle arbitrary types.
        """
    constants = {'scalar_const', 'dense_const', 'sparse_const'}
    if not column and lin_op.type in constants and (len(lin_op.shape) == 2):
        constant_data = self.get_constant_data_from_const(lin_op)
        return (constant_data, True)
    constant_view = self.process_constraint(lin_op, view)
    assert constant_view.variable_ids == {Constant.ID.value}
    constant_data = constant_view.tensor[Constant.ID.value]
    if not column and len(lin_op.shape) >= 1:
        lin_op_shape = lin_op.shape if len(lin_op.shape) == 2 else [1, lin_op.shape[0]]
        constant_data = self.reshape_constant_data(constant_data, lin_op_shape)
    data_to_return = constant_data[Constant.ID.value] if constant_view.is_parameter_free else constant_data
    return (data_to_return, constant_view.is_parameter_free)