from __future__ import annotations
from typing import Union
import numpy as np
def make_ry(phi: float, out: np.ndarray) -> np.ndarray:
    """
    Makes a 2x2 matrix that corresponds to Y-rotation gate.
    This is a fast implementation that does not allocate the output matrix.

    Args:
        phi: rotation angle.
        out: placeholder for the result (2x2, complex-valued matrix).

    Returns:
        rotation gate, same object as referenced by "out".
    """
    a = 0.5 * phi
    cs, sn = (np.cos(a).item(), np.sin(a).item())
    out[0, 0] = cs
    out[0, 1] = -sn
    out[1, 0] = sn
    out[1, 1] = cs
    return out