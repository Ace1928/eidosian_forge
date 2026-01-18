import math
import numpy as np
from .casting import sctypes
def angle_axis2mat(theta, vector, is_normalized=False):
    """Rotation matrix of angle `theta` around `vector`

    Parameters
    ----------
    theta : scalar
       angle of rotation
    vector : 3 element sequence
       vector specifying axis for rotation.
    is_normalized : bool, optional
       True if vector is already normalized (has norm of 1).  Default
       False

    Returns
    -------
    mat : array shape (3,3)
       rotation matrix for specified rotation

    Notes
    -----
    From: https://en.wikipedia.org/wiki/Rotation_matrix#Axis_and_angle
    """
    x, y, z = vector
    if not is_normalized:
        n = math.sqrt(x * x + y * y + z * z)
        x = x / n
        y = y / n
        z = z / n
    c, s = (math.cos(theta), math.sin(theta))
    C = 1 - c
    xs, ys, zs = (x * s, y * s, z * s)
    xC, yC, zC = (x * C, y * C, z * C)
    xyC, yzC, zxC = (x * yC, y * zC, z * xC)
    return np.array([[x * xC + c, xyC - zs, zxC + ys], [xyC + zs, y * yC + c, yzC - xs], [zxC - ys, yzC + xs, z * zC + c]])