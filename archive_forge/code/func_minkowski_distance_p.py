import numpy as np
from ._ckdtree import cKDTree, cKDTreeNode
def minkowski_distance_p(x, y, p=2):
    """Compute the pth power of the L**p distance between two arrays.

    For efficiency, this function computes the L**p distance but does
    not extract the pth root. If `p` is 1 or infinity, this is equal to
    the actual L**p distance.

    The last dimensions of `x` and `y` must be the same length.  Any
    other dimensions must be compatible for broadcasting.

    Parameters
    ----------
    x : (..., K) array_like
        Input array.
    y : (..., K) array_like
        Input array.
    p : float, 1 <= p <= infinity
        Which Minkowski p-norm to use.

    Returns
    -------
    dist : ndarray
        pth power of the distance between the input arrays.

    Examples
    --------
    >>> from scipy.spatial import minkowski_distance_p
    >>> minkowski_distance_p([[0, 0], [0, 0]], [[1, 1], [0, 1]])
    array([2, 1])

    """
    x = np.asarray(x)
    y = np.asarray(y)
    common_datatype = np.promote_types(np.promote_types(x.dtype, y.dtype), 'float64')
    x = x.astype(common_datatype)
    y = y.astype(common_datatype)
    if p == np.inf:
        return np.amax(np.abs(y - x), axis=-1)
    elif p == 1:
        return np.sum(np.abs(y - x), axis=-1)
    else:
        return np.sum(np.abs(y - x) ** p, axis=-1)