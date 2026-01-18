import warnings
import bisect
from collections import deque
import numpy as np
from . import _hierarchy, _optimal_leaf_ordering
import scipy.spatial.distance as distance
from scipy._lib._array_api import array_namespace, as_xparray, copy
from scipy._lib._disjoint_set import DisjointSet
def maxinconsts(Z, R):
    """
    Return the maximum inconsistency coefficient for each
    non-singleton cluster and its children.

    Parameters
    ----------
    Z : ndarray
        The hierarchical clustering encoded as a matrix. See
        `linkage` for more information.
    R : ndarray
        The inconsistency matrix.

    Returns
    -------
    MI : ndarray
        A monotonic ``(n-1)``-sized numpy array of doubles.

    See Also
    --------
    linkage : for a description of what a linkage matrix is.
    inconsistent : for the creation of a inconsistency matrix.

    Examples
    --------
    >>> from scipy.cluster.hierarchy import median, inconsistent, maxinconsts
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
    >>> Z
    array([[ 0.        ,  1.        ,  1.        ,  2.        ],
           [ 3.        ,  4.        ,  1.        ,  2.        ],
           [ 9.        , 10.        ,  1.        ,  2.        ],
           [ 6.        ,  7.        ,  1.        ,  2.        ],
           [ 2.        , 12.        ,  1.11803399,  3.        ],
           [ 5.        , 13.        ,  1.11803399,  3.        ],
           [ 8.        , 15.        ,  1.11803399,  3.        ],
           [11.        , 14.        ,  1.11803399,  3.        ],
           [18.        , 19.        ,  3.        ,  6.        ],
           [16.        , 17.        ,  3.5       ,  6.        ],
           [20.        , 21.        ,  3.25      , 12.        ]])
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

    Here, `scipy.cluster.hierarchy.maxinconsts` can be used to compute
    the maximum value of the inconsistency statistic (the last column of
    ``R``) for each non-singleton cluster and its children:

    >>> maxinconsts(Z, R)
    array([0.        , 0.        , 0.        , 0.        , 0.70710678,
           0.70710678, 0.70710678, 0.70710678, 1.15470054, 1.15470054,
           1.15470054])

    """
    xp = array_namespace(Z, R)
    Z = as_xparray(Z, order='C', dtype=xp.float64, xp=xp)
    R = as_xparray(R, order='C', dtype=xp.float64, xp=xp)
    is_valid_linkage(Z, throw=True, name='Z')
    is_valid_im(R, throw=True, name='R')
    n = Z.shape[0] + 1
    if Z.shape[0] != R.shape[0]:
        raise ValueError('The inconsistency matrix and linkage matrix each have a different number of rows.')
    MI = np.zeros((n - 1,))
    Z = np.asarray(Z)
    R = np.asarray(R)
    _hierarchy.get_max_Rfield_for_each_cluster(Z, R, MI, int(n), 3)
    MI = xp.asarray(MI)
    return MI