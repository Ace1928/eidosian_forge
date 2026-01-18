import numpy as np  # type: ignore
from typing import Tuple, Optional
def homog_scale_mtx(scale: float) -> np.array:
    """Generate a 4x4 NumPy scaling matrix.

    :param float scale: scale multiplier
    """
    return np.array([[scale, 0, 0, 0], [0, scale, 0, 0], [0, 0, scale, 0], [0, 0, 0, 1]], dtype=np.float64)