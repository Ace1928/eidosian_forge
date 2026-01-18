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
def process_constraint(self, lin_op: LinOp, empty_view: TensorView) -> TensorView:
    """
        Depth-first parsing of a linOp node.

        Parameters
        ----------
        lin_op: a node in the linOp tree.
        empty_view: TensorView used to create tensors for leaf nodes.

        Returns
        -------
        The processed node as a TensorView.
        """
    if lin_op.type == 'variable':
        assert isinstance(lin_op.data, int)
        assert len(lin_op.shape) in {0, 1, 2}
        variable_tensor = self.get_variable_tensor(lin_op.shape, lin_op.data)
        return empty_view.create_new_tensor_view({lin_op.data}, variable_tensor, is_parameter_free=True)
    elif lin_op.type in {'scalar_const', 'dense_const', 'sparse_const'}:
        data_tensor = self.get_data_tensor(lin_op.data)
        return empty_view.create_new_tensor_view({Constant.ID.value}, data_tensor, is_parameter_free=True)
    elif lin_op.type == 'param':
        param_tensor = self.get_param_tensor(lin_op.shape, lin_op.data)
        return empty_view.create_new_tensor_view({Constant.ID.value}, param_tensor, is_parameter_free=False)
    else:
        func = self.get_func(lin_op.type)
        if lin_op.type in {'vstack', 'hstack'}:
            return func(lin_op, empty_view)
        res = None
        for arg in lin_op.args:
            arg_coeff = self.process_constraint(arg, empty_view)
            arg_res = func(lin_op, arg_coeff)
            if res is None:
                res = arg_res
            else:
                res += arg_res
        assert res is not None
        return res