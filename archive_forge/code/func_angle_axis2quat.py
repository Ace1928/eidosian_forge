import math
import numpy as np
from .casting import sctypes
def angle_axis2quat(theta, vector, is_normalized=False):
    """Quaternion for rotation of angle `theta` around `vector`

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
    quat : 4 element sequence of symbols
       quaternion giving specified rotation

    Examples
    --------
    >>> q = angle_axis2quat(np.pi, [1, 0, 0])
    >>> np.allclose(q, [0, 1, 0,  0])
    True

    Notes
    -----
    Formula from http://mathworld.wolfram.com/EulerParameters.html
    """
    vector = np.array(vector)
    if not is_normalized:
        vector = vector / math.sqrt(np.dot(vector, vector))
    t2 = theta / 2.0
    st2 = math.sin(t2)
    return np.concatenate(([math.cos(t2)], vector * st2))