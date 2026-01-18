import numpy as np  # type: ignore
from typing import Tuple, Optional
def multi_rot_Y(angle_rads: np.ndarray) -> np.ndarray:
    """Create [entries] NumPy Y rotation matrices for [entries] angles.

    :param entries: int number of matrices generated.
    :param angle_rads: NumPy array of angles
    :returns: entries x 4 x 4 homogeneous rotation matrices
    """
    ry = np.empty((angle_rads.shape[0], 4, 4))
    ry[...] = np.identity(4)
    ry[:, 0, 0] = ry[:, 2, 2] = np.cos(angle_rads)
    ry[:, 0, 2] = np.sin(angle_rads)
    ry[:, 2, 0] = -ry[:, 0, 2]
    return ry