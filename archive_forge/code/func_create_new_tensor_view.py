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
def create_new_tensor_view(self, variable_ids: set[int], tensor: Any, is_parameter_free: bool) -> SciPyTensorView:
    """
        Create new SciPyTensorView with same shape information as self,
        but new tensor data.
        """
    return SciPyTensorView(variable_ids, tensor, is_parameter_free, self.param_size_plus_one, self.id_to_col, self.param_to_size, self.param_to_col, self.var_length)