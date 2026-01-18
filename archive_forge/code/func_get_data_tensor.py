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
def get_data_tensor(self, data: np.ndarray | sp.spmatrix) -> dict[int, dict[int, sp.csr_matrix]]:
    """
        Returns tensor of constant node as a column vector.
        This function reshapes the data and converts it to csc format.
        """
    if isinstance(data, np.ndarray):
        tensor = sp.csr_matrix(data.reshape((-1, 1), order='F'))
    else:
        tensor = sp.coo_matrix(data).reshape((-1, 1), order='F').tocsr()
    return {Constant.ID.value: {Constant.ID.value: tensor}}