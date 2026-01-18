from __future__ import annotations
from typing import Union
import numpy as np
def make_rz(phi: float, out: np.ndarray) -> np.ndarray:
    """
    Makes a 2x2 matrix that corresponds to Z-rotation gate.
    This is a fast implementation that does not allocate the output matrix.

    Args:
        phi: rotation angle.
        out: placeholder for the result (2x2, complex-valued matrix).

    Returns:
        rotation gate, same object as referenced by "out".
    """
    exp = np.exp(0.5j * phi).item()
    out[0, 0] = 1.0 / exp
    out[0, 1] = 0
    out[1, 0] = 0
    out[1, 1] = exp
    return out