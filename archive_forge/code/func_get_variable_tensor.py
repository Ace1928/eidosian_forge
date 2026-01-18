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
def get_variable_tensor(self, shape: tuple[int, ...], variable_id: int) -> dict[int, dict[int, sp.csc_matrix]]:
    """
        Returns tensor of a variable node, i.e., eye(n) across axes 0 and 1, where n is
        the size of the variable.
        This function returns eye(n) in csc format.
        """
    assert variable_id != Constant.ID
    n = int(np.prod(shape))
    return {variable_id: {Constant.ID.value: sp.eye(n, format='csc')}}