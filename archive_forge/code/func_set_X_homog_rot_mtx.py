import numpy as np  # type: ignore
from typing import Tuple, Optional
def set_X_homog_rot_mtx(angle_rads: float, mtx: np.ndarray):
    """Update existing X rotation matrix to new angle."""
    cosang = np.cos(angle_rads)
    sinang = np.sin(angle_rads)
    mtx[1][1] = mtx[2][2] = cosang
    mtx[2][1] = sinang
    mtx[1][2] = -sinang