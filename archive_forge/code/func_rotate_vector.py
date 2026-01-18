import math
import numpy as np
from .casting import sctypes
def rotate_vector(v, q):
    """Apply transformation in quaternion `q` to vector `v`

    Parameters
    ----------
    v : 3 element sequence
       3 dimensional vector
    q : 4 element sequence
       w, i, j, k of quaternion

    Returns
    -------
    vdash : array shape (3,)
       `v` rotated by quaternion `q`

    Notes
    -----
    See:
    https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Describing_rotations_with_quaternions

    """
    varr = np.zeros((4,))
    varr[1:] = v
    return mult(q, mult(varr, conjugate(q)))[1:]