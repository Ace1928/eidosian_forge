import math
import numpy as np
from .casting import sctypes
def nearly_equivalent(q1, q2, rtol=1e-05, atol=1e-08):
    """Returns True if `q1` and `q2` give near equivalent transforms

    `q1` may be nearly numerically equal to `q2`, or nearly equal to `q2` * -1
    (because a quaternion multiplied by -1 gives the same transform).

    Parameters
    ----------
    q1 : 4 element sequence
       w, x, y, z of first quaternion
    q2 : 4 element sequence
       w, x, y, z of second quaternion

    Returns
    -------
    equiv : bool
       True if `q1` and `q2` are nearly equivalent, False otherwise

    Examples
    --------
    >>> q1 = [1, 0, 0, 0]
    >>> nearly_equivalent(q1, [0, 1, 0, 0])
    False
    >>> nearly_equivalent(q1, [1, 0, 0, 0])
    True
    >>> nearly_equivalent(q1, [-1, 0, 0, 0])
    True
    """
    q1 = np.array(q1)
    q2 = np.array(q2)
    if np.allclose(q1, q2, rtol, atol):
        return True
    return np.allclose(q1 * -1, q2, rtol, atol)