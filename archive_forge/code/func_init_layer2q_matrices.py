from __future__ import annotations
from abc import abstractmethod, ABC
from typing import Optional
import numpy as np
from .fast_grad_utils import (
def init_layer2q_matrices(thetas: np.ndarray, dst: np.ndarray) -> np.ndarray:
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
    depth = thetas.shape[0]
    for k in range(depth):
        th = thetas[k]
        cs0 = np.cos(0.5 * th[0]).item()
        sn0 = np.sin(0.5 * th[0]).item()
        ep1 = np.exp(0.5j * th[1]).item()
        en1 = np.exp(-0.5j * th[1]).item()
        cs2 = np.cos(0.5 * th[2]).item()
        sn2 = np.sin(0.5 * th[2]).item()
        cs3 = np.cos(0.5 * th[3]).item()
        sn3 = np.sin(0.5 * th[3]).item()
        ep1cs0 = ep1 * cs0
        ep1sn0 = ep1 * sn0
        en1cs0 = en1 * cs0
        en1sn0 = en1 * sn0
        sn2cs3 = sn2 * cs3
        sn2sn3 = sn2 * sn3
        cs2cs3 = cs2 * cs3
        sn3cs2j = 1j * sn3 * cs2
        sn2sn3j = 1j * sn2sn3
        flat_dst = dst[k].ravel()
        flat_dst[:] = [-(sn2sn3j - cs2cs3) * en1cs0, -(sn2cs3 + sn3cs2j) * en1cs0, (sn2cs3 + sn3cs2j) * en1sn0, (sn2sn3j - cs2cs3) * en1sn0, (sn2cs3 - sn3cs2j) * en1cs0, (sn2sn3j + cs2cs3) * en1cs0, -(sn2sn3j + cs2cs3) * en1sn0, (-sn2cs3 + sn3cs2j) * en1sn0, (-sn2sn3j + cs2cs3) * ep1sn0, -(sn2cs3 + sn3cs2j) * ep1sn0, -(sn2cs3 + sn3cs2j) * ep1cs0, (-sn2sn3j + cs2cs3) * ep1cs0, (sn2cs3 - sn3cs2j) * ep1sn0, (sn2sn3j + cs2cs3) * ep1sn0, (sn2sn3j + cs2cs3) * ep1cs0, (sn2cs3 - sn3cs2j) * ep1cs0]
    return dst