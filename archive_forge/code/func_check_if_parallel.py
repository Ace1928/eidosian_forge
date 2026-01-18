from functools import lru_cache
import math
import warnings
import numpy as np
from matplotlib import _api
def check_if_parallel(dx1, dy1, dx2, dy2, tolerance=1e-05):
    """
    Check if two lines are parallel.

    Parameters
    ----------
    dx1, dy1, dx2, dy2 : float
        The gradients *dy*/*dx* of the two lines.
    tolerance : float
        The angular tolerance in radians up to which the lines are considered
        parallel.

    Returns
    -------
    is_parallel
        - 1 if two lines are parallel in same direction.
        - -1 if two lines are parallel in opposite direction.
        - False otherwise.
    """
    theta1 = np.arctan2(dx1, dy1)
    theta2 = np.arctan2(dx2, dy2)
    dtheta = abs(theta1 - theta2)
    if dtheta < tolerance:
        return 1
    elif abs(dtheta - np.pi) < tolerance:
        return -1
    else:
        return False