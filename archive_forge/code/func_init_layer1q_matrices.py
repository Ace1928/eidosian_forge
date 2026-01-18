from __future__ import annotations
from abc import abstractmethod, ABC
from typing import Optional
import numpy as np
from .fast_grad_utils import (
def init_layer1q_matrices(thetas: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """
    Initializes 4x4 matrices of 2-qubit gates defined in the paper.

    Args:
        thetas: depth x 4 matrix of gate parameters for every layer, where
                "depth" is the number of layers.
        dst: destination array of size depth x 4 x 4 that will receive gate
             matrices of each layer.

    Returns:
        Returns the "dst" array.
    """
    n = thetas.shape[0]
    tmp = np.full((4, 2, 2), fill_value=0, dtype=np.complex128)
    for k in range(n):
        th = thetas[k]
        a = make_rz(th[0], out=tmp[0])
        b = make_ry(th[1], out=tmp[1])
        c = make_rz(th[2], out=tmp[2])
        np.dot(np.dot(a, b, out=tmp[3]), c, out=dst[k])
    return dst