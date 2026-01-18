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
@classmethod
def get_empty_view(cls, param_size_plus_one: int, id_to_col: dict[int, int], param_to_size: dict[int, int], param_to_col: dict[int, int], var_length: int) -> TensorView:
    """
        Return a TensorView that has shape information, but no data.
        """
    return cls(None, None, True, param_size_plus_one, id_to_col, param_to_size, param_to_col, var_length)