import numpy as np  # type: ignore
from typing import Tuple, Optional
def set_homog_trans_mtx(x: float, y: float, z: float, mtx: np.ndarray):
    """Update existing translation matrix to new values."""
    mtx[0][3] = x
    mtx[1][3] = y
    mtx[2][3] = z