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
def combine_potentially_none(self, a: dict | None, b: dict | None) -> dict | None:
    """
        Adds the tensor a to b if they are both not none.
        If a (b) is not None but b (a) is None, returns a (b).
        Returns None if both a and b are None.
        """
    if a is None and b is None:
        return None
    elif a is not None and b is None:
        return a
    elif a is None and b is not None:
        return b
    else:
        return self.add_dicts(a, b)