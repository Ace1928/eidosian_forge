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
def concatenate_tensors(self, tensors: list[TensorRepresentation]) -> TensorRepresentation:
    """
        Takes list of tensors which have already been offset along axis 0 (rows) and
        combines them into a single tensor.
        """
    return TensorRepresentation.combine(tensors)