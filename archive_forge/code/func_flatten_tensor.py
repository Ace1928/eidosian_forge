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
def flatten_tensor(self, num_param_slices: int) -> sp.csc_matrix:
    """
        Flatten into 2D scipy csc-matrix in column-major order and transpose.
        """
    rows = self.col.astype(np.int64) * np.int64(self.shape[0]) + self.row.astype(np.int64)
    cols = self.parameter_offset.astype(np.int64)
    shape = (np.int64(np.prod(self.shape)), num_param_slices)
    return sp.csc_matrix((self.data, (rows, cols)), shape=shape)