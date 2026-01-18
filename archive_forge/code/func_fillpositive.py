import math
import numpy as np
from .casting import sctypes
def fillpositive(xyz, w2_thresh=None):
    """Compute unit quaternion from last 3 values

    Parameters
    ----------
    xyz : iterable
       iterable containing 3 values, corresponding to quaternion x, y, z
    w2_thresh : None or float, optional
       threshold to determine if w squared is non-zero.
       If None (default) then w2_thresh set equal to
       3 * ``np.finfo(xyz.dtype).eps``, if possible, otherwise
       3 * ``np.finfo(np.float64).eps``

    Returns
    -------
    wxyz : array shape (4,)
         Full 4 values of quaternion

    Notes
    -----
    If w, x, y, z are the values in the full quaternion, assumes w is
    positive.

    Gives error if w*w is estimated to be negative

    w = 0 corresponds to a 180 degree rotation

    The unit quaternion specifies that np.dot(wxyz, wxyz) == 1.

    If w is positive (assumed here), w is given by:

    w = np.sqrt(1.0-(x*x+y*y+z*z))

    w2 = 1.0-(x*x+y*y+z*z) can be near zero, which will lead to
    numerical instability in sqrt.  Here we use the system maximum
    float type to reduce numerical instability

    Examples
    --------
    >>> import numpy as np
    >>> wxyz = fillpositive([0,0,0])
    >>> np.all(wxyz == [1, 0, 0, 0])
    True
    >>> wxyz = fillpositive([1,0,0]) # Corner case; w is 0
    >>> np.all(wxyz == [0, 1, 0, 0])
    True
    >>> np.dot(wxyz, wxyz)
    1.0
    """
    if len(xyz) != 3:
        raise ValueError('xyz should have length 3')
    if w2_thresh is None:
        try:
            w2_thresh = np.finfo(xyz.dtype).eps * 3
        except (AttributeError, ValueError):
            w2_thresh = FLOAT_EPS * 3
    xyz = np.asarray(xyz, dtype=MAX_FLOAT)
    w2 = 1.0 - xyz @ xyz
    if np.abs(w2) < np.abs(w2_thresh):
        w = 0
    elif w2 < 0:
        raise ValueError(f'w2 should be positive, but is {w2:e}')
    else:
        w = np.sqrt(w2)
    return np.r_[w, xyz]