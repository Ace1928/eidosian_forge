import numpy as np  # type: ignore
from typing import Tuple, Optional
def multi_rot_Z(angle_rads: np.ndarray) -> np.ndarray:
    """Create [entries] NumPy Z rotation matrices for [entries] angles.

    :param entries: int number of matrices generated.
    :param angle_rads: NumPy array of angles
    :returns: entries x 4 x 4 homogeneous rotation matrices
    """
    rz = np.empty((angle_rads.shape[0], 4, 4))
    rz[...] = np.identity(4)
    rz[:, 0, 0] = rz[:, 1, 1] = np.cos(angle_rads)
    rz[:, 1, 0] = np.sin(angle_rads)
    rz[:, 0, 1] = -rz[:, 1, 0]
    return rz