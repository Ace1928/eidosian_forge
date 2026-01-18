from __future__ import annotations
from abc import abstractmethod, ABC
from typing import Optional
import numpy as np
from .fast_grad_utils import (
def set_from_matrix(self, mat: np.ndarray):
    """See base class description."""
    np.copyto(self._gmat, mat)