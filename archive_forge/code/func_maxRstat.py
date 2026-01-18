import warnings
import bisect
from collections import deque
import numpy as np
from . import _hierarchy, _optimal_leaf_ordering
import scipy.spatial.distance as distance
from scipy._lib._array_api import array_namespace, as_xparray, copy
from scipy._lib._disjoint_set import DisjointSet
def maxRstat(Z, R, i):
    """
    Return the maximum statistic for each non-singleton cluster and its
    children.

    Parameters
    ----------
    Z : array_like
        The hierarchical clustering encoded as a matrix. See `linkage` for more
        information.
    R : array_like
        The inconsistency matrix.
    i : int
        The column of `R` to use as the statistic.

    Returns
    -------
    MR : ndarray
        Calculates the maximum statistic for the i'th column of the
        inconsistency matrix `R` for each non-singleton cluster
        node. ``MR[j]`` is the maximum over ``R[Q(j)-n, i]``, where
        ``Q(j)`` the set of all node ids corresponding to nodes below
        and including ``j``.

    See Also
    --------
    linkage : for a description of what a linkage matrix is.
    inconsistent : for the creation of a inconsistency matrix.

    Examples
    --------
    >>> from scipy.cluster.hierarchy import median, inconsistent, maxRstat
    >>> from scipy.spatial.distance import pdist

    Given a data set ``X``, we can apply a clustering method to obtain a
    linkage matrix ``Z``. `scipy.cluster.hierarchy.inconsistent` can
    be also used to obtain the inconsistency matrix ``R`` associated to
    this clustering process:

    >>> X = [[0, 0], [0, 1], [1, 0],
    ...      [0, 4], [0, 3], [1, 4],
    ...      [4, 0], [3, 0], [4, 1],
    ...      [4, 4], [3, 4], [4, 3]]

    >>> Z = median(pdist(X))
    >>> R = inconsistent(Z)
    >>> R
    array([[1.        , 0.        , 1.        , 0.        ],
           [1.        , 0.        , 1.        , 0.        ],
           [1.        , 0.        , 1.        , 0.        ],
           [1.        , 0.        , 1.        , 0.        ],
           [1.05901699, 0.08346263, 2.        , 0.70710678],
           [1.05901699, 0.08346263, 2.        , 0.70710678],
           [1.05901699, 0.08346263, 2.        , 0.70710678],
           [1.05901699, 0.08346263, 2.        , 0.70710678],
           [1.74535599, 1.08655358, 3.        , 1.15470054],
           [1.91202266, 1.37522872, 3.        , 1.15470054],
           [3.25      , 0.25      , 3.        , 0.        ]])

    `scipy.cluster.hierarchy.maxRstat` can be used to compute
    the maximum value of each column of ``R``, for each non-singleton
    cluster and its children:

    >>> maxRstat(Z, R, 0)
    array([1.        , 1.        , 1.        , 1.        , 1.05901699,
           1.05901699, 1.05901699, 1.05901699, 1.74535599, 1.91202266,
           3.25      ])
    >>> maxRstat(Z, R, 1)
    array([0.        , 0.        , 0.        , 0.        , 0.08346263,
           0.08346263, 0.08346263, 0.08346263, 1.08655358, 1.37522872,
           1.37522872])
    >>> maxRstat(Z, R, 3)
    array([0.        , 0.        , 0.        , 0.        , 0.70710678,
           0.70710678, 0.70710678, 0.70710678, 1.15470054, 1.15470054,
           1.15470054])

    """
    xp = array_namespace(Z, R)
    Z = as_xparray(Z, order='C', dtype=xp.float64, xp=xp)
    R = as_xparray(R, order='C', dtype=xp.float64, xp=xp)
    is_valid_linkage(Z, throw=True, name='Z')
    is_valid_im(R, throw=True, name='R')
    if not isinstance(i, int):
        raise TypeError('The third argument must be an integer.')
    if i < 0 or i > 3:
        raise ValueError('i must be an integer between 0 and 3 inclusive.')
    if Z.shape[0] != R.shape[0]:
        raise ValueError('The inconsistency matrix and linkage matrix each have a different number of rows.')
    n = Z.shape[0] + 1
    MR = np.zeros((n - 1,))
    Z = np.asarray(Z)
    R = np.asarray(R)
    _hierarchy.get_max_Rfield_for_each_cluster(Z, R, MR, int(n), i)
    MR = xp.asarray(MR)
    return MR