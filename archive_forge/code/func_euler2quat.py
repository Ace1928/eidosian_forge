import math
from functools import reduce
import numpy as np
def euler2quat(z=0, y=0, x=0):
    """Return quaternion corresponding to these Euler angles

    Uses the z, then y, then x convention above

    Parameters
    ----------
    z : scalar
       Rotation angle in radians around z-axis (performed first)
    y : scalar
       Rotation angle in radians around y-axis
    x : scalar
       Rotation angle in radians around x-axis (performed last)

    Returns
    -------
    quat : array shape (4,)
       Quaternion in w, x, y z (real, then vector) format

    Notes
    -----
    We can derive this formula in Sympy using:

    1. Formula giving quaternion corresponding to rotation of theta radians
       about arbitrary axis:
       http://mathworld.wolfram.com/EulerParameters.html
    2. Generated formulae from 1.) for quaternions corresponding to
       theta radians rotations about ``x, y, z`` axes
    3. Apply quaternion multiplication formula -
       https://en.wikipedia.org/wiki/Quaternions#Hamilton_product - to
       formulae from 2.) to give formula for combined rotations.
    """
    z = z / 2.0
    y = y / 2.0
    x = x / 2.0
    cz = math.cos(z)
    sz = math.sin(z)
    cy = math.cos(y)
    sy = math.sin(y)
    cx = math.cos(x)
    sx = math.sin(x)
    return np.array([cx * cy * cz - sx * sy * sz, cx * sy * sz + cy * cz * sx, cx * cz * sy - sx * cy * sz, cx * cy * sz + sx * cz * sy])