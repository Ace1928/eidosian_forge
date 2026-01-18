import warnings
import bisect
from collections import deque
import numpy as np
from . import _hierarchy, _optimal_leaf_ordering
import scipy.spatial.distance as distance
from scipy._lib._array_api import array_namespace, as_xparray, copy
from scipy._lib._disjoint_set import DisjointSet
def is_valid_im(R, warning=False, throw=False, name=None):
    """Return True if the inconsistency matrix passed is valid.

    It must be a :math:`n` by 4 array of doubles. The standard
    deviations ``R[:,1]`` must be nonnegative. The link counts
    ``R[:,2]`` must be positive and no greater than :math:`n-1`.

    Parameters
    ----------
    R : ndarray
        The inconsistency matrix to check for validity.
    warning : bool, optional
        When True, issues a Python warning if the linkage
        matrix passed is invalid.
    throw : bool, optional
        When True, throws a Python exception if the linkage
        matrix passed is invalid.
    name : str, optional
        This string refers to the variable name of the invalid
        linkage matrix.

    Returns
    -------
    b : bool
        True if the inconsistency matrix is valid.

    See Also
    --------
    linkage : for a description of what a linkage matrix is.
    inconsistent : for the creation of a inconsistency matrix.

    Examples
    --------
    >>> from scipy.cluster.hierarchy import ward, inconsistent, is_valid_im
    >>> from scipy.spatial.distance import pdist

    Given a data set ``X``, we can apply a clustering method to obtain a
    linkage matrix ``Z``. `scipy.cluster.hierarchy.inconsistent` can
    be also used to obtain the inconsistency matrix ``R`` associated to
    this clustering process:

    >>> X = [[0, 0], [0, 1], [1, 0],
    ...      [0, 4], [0, 3], [1, 4],
    ...      [4, 0], [3, 0], [4, 1],
    ...      [4, 4], [3, 4], [4, 3]]

    >>> Z = ward(pdist(X))
    >>> R = inconsistent(Z)
    >>> Z
    array([[ 0.        ,  1.        ,  1.        ,  2.        ],
           [ 3.        ,  4.        ,  1.        ,  2.        ],
           [ 6.        ,  7.        ,  1.        ,  2.        ],
           [ 9.        , 10.        ,  1.        ,  2.        ],
           [ 2.        , 12.        ,  1.29099445,  3.        ],
           [ 5.        , 13.        ,  1.29099445,  3.        ],
           [ 8.        , 14.        ,  1.29099445,  3.        ],
           [11.        , 15.        ,  1.29099445,  3.        ],
           [16.        , 17.        ,  5.77350269,  6.        ],
           [18.        , 19.        ,  5.77350269,  6.        ],
           [20.        , 21.        ,  8.16496581, 12.        ]])
    >>> R
    array([[1.        , 0.        , 1.        , 0.        ],
           [1.        , 0.        , 1.        , 0.        ],
           [1.        , 0.        , 1.        , 0.        ],
           [1.        , 0.        , 1.        , 0.        ],
           [1.14549722, 0.20576415, 2.        , 0.70710678],
           [1.14549722, 0.20576415, 2.        , 0.70710678],
           [1.14549722, 0.20576415, 2.        , 0.70710678],
           [1.14549722, 0.20576415, 2.        , 0.70710678],
           [2.78516386, 2.58797734, 3.        , 1.15470054],
           [2.78516386, 2.58797734, 3.        , 1.15470054],
           [6.57065706, 1.38071187, 3.        , 1.15470054]])

    Now we can use `scipy.cluster.hierarchy.is_valid_im` to verify that
    ``R`` is correct:

    >>> is_valid_im(R)
    True

    However, if ``R`` is wrongly constructed (e.g., one of the standard
    deviations is set to a negative value), then the check will fail:

    >>> R[-1,1] = R[-1,1] * -1
    >>> is_valid_im(R)
    False

    """
    xp = array_namespace(R)
    R = as_xparray(R, order='c', xp=xp)
    valid = True
    name_str = '%r ' % name if name else ''
    try:
        if R.dtype != xp.float64:
            raise TypeError('Inconsistency matrix %smust contain doubles (double).' % name_str)
        if len(R.shape) != 2:
            raise ValueError('Inconsistency matrix %smust have shape=2 (i.e. be two-dimensional).' % name_str)
        if R.shape[1] != 4:
            raise ValueError('Inconsistency matrix %smust have 4 columns.' % name_str)
        if R.shape[0] < 1:
            raise ValueError('Inconsistency matrix %smust have at least one row.' % name_str)
        if xp.any(R[:, 0] < 0):
            raise ValueError('Inconsistency matrix %scontains negative link height means.' % name_str)
        if xp.any(R[:, 1] < 0):
            raise ValueError('Inconsistency matrix %scontains negative link height standard deviations.' % name_str)
        if xp.any(R[:, 2] < 0):
            raise ValueError('Inconsistency matrix %scontains negative link counts.' % name_str)
    except Exception as e:
        if throw:
            raise
        if warning:
            _warning(str(e))
        valid = False
    return valid