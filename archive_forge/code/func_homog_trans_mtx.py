import numpy as np  # type: ignore
from typing import Tuple, Optional
def homog_trans_mtx(x: float, y: float, z: float) -> np.array:
    """Generate a 4x4 NumPy translation matrix.

    :param x, y, z: translation in each axis
    """
    return np.array(((1, 0, 0, x), (0, 1, 0, y), (0, 0, 1, z), (0, 0, 0, 1)), dtype=np.float64)