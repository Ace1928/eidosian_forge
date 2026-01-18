import numpy as np  # type: ignore
from typing import Tuple, Optional
def set_Z_homog_rot_mtx(angle_rads: float, mtx: np.ndarray):
    """Update existing Z rotation matrix to new angle."""
    cosang = np.cos(angle_rads)
    sinang = np.sin(angle_rads)
    mtx[0][0] = mtx[1][1] = cosang
    mtx[1][0] = sinang
    mtx[0][1] = -sinang