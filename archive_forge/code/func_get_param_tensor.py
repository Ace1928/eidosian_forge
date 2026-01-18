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
def get_param_tensor(self, shape: tuple[int, ...], parameter_id: int) -> dict[int, dict[int, sp.csc_matrix]]:
    """
        Returns tensor of a parameter node, i.e., eye(n) across axes 0 and 2, where n is
        the size of the parameter.
        This function returns eye(n).flatten() in csc format.
        """
    assert parameter_id != Constant.ID
    param_size = self.param_to_size[parameter_id]
    shape = (int(np.prod(shape) * param_size), 1)
    arg = (np.ones(param_size), (np.arange(param_size) + np.arange(param_size) * param_size, np.zeros(param_size)))
    param_vec = sp.csc_matrix(arg, shape)
    return {Constant.ID.value: {parameter_id: param_vec}}