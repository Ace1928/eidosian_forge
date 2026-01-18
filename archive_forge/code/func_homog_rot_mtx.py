import numpy as np  # type: ignore
from typing import Tuple, Optional
def homog_rot_mtx(angle_rads: float, axis: str) -> np.array:
    """Generate a 4x4 single-axis NumPy rotation matrix.

    :param float angle_rads: the desired rotation angle in radians
    :param char axis: character specifying the rotation axis
    """
    cosang = np.cos(angle_rads)
    sinang = np.sin(angle_rads)
    if 'z' == axis:
        return np.array(((cosang, -sinang, 0, 0), (sinang, cosang, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)), dtype=np.float64)
    elif 'y' == axis:
        return np.array(((cosang, 0, sinang, 0), (0, 1, 0, 0), (-sinang, 0, cosang, 0), (0, 0, 0, 1)), dtype=np.float64)
    else:
        return np.array(((1, 0, 0, 0), (0, cosang, -sinang, 0), (0, sinang, cosang, 0), (0, 0, 0, 1)), dtype=np.float64)