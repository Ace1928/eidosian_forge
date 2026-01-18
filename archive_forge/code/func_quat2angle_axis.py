import math
import numpy as np
from .casting import sctypes
def quat2angle_axis(quat, identity_thresh=None):
    """Convert quaternion to rotation of angle around axis

    Parameters
    ----------
    quat : 4 element sequence
       w, x, y, z forming quaternion
    identity_thresh : None or scalar, optional
       threshold below which the norm of the vector part of the
       quaternion (x, y, z) is deemed to be 0, leading to the identity
       rotation.  None (the default) leads to a threshold estimated
       based on the precision of the input.

    Returns
    -------
    theta : scalar
       angle of rotation
    vector : array shape (3,)
       axis around which rotation occurs

    Examples
    --------
    >>> theta, vec = quat2angle_axis([0, 1, 0, 0])
    >>> np.allclose(theta, np.pi)
    True
    >>> vec
    array([1., 0., 0.])

    If this is an identity rotation, we return a zero angle and an
    arbitrary vector

    >>> quat2angle_axis([1, 0, 0, 0])
    (0.0, array([1., 0., 0.]))

    Notes
    -----
    A quaternion for which x, y, z are all equal to 0, is an identity
    rotation.  In this case we return a 0 angle and an  arbitrary
    vector, here [1, 0, 0]
    """
    w, x, y, z = quat
    vec = np.asarray([x, y, z])
    if identity_thresh is None:
        try:
            identity_thresh = np.finfo(vec.dtype).eps * 3
        except ValueError:
            identity_thresh = FLOAT_EPS * 3
    n = math.sqrt(x * x + y * y + z * z)
    if n < identity_thresh:
        return (0.0, np.array([1.0, 0, 0]))
    return (2 * math.acos(w), vec / n)