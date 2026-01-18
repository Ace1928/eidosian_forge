import warnings
import bisect
from collections import deque
import numpy as np
from . import _hierarchy, _optimal_leaf_ordering
import scipy.spatial.distance as distance
from scipy._lib._array_api import array_namespace, as_xparray, copy
from scipy._lib._disjoint_set import DisjointSet
def inconsistent(Z, d=2):
    """
    Calculate inconsistency statistics on a linkage matrix.

    Parameters
    ----------
    Z : ndarray
        The :math:`(n-1)` by 4 matrix encoding the linkage (hierarchical
        clustering).  See `linkage` documentation for more information on its
        form.
    d : int, optional
        The number of links up to `d` levels below each non-singleton cluster.

    Returns
    -------
    R : ndarray
        A :math:`(n-1)` by 4 matrix where the ``i``'th row contains the link
        statistics for the non-singleton cluster ``i``. The link statistics are
        computed over the link heights for links :math:`d` levels below the
        cluster ``i``. ``R[i,0]`` and ``R[i,1]`` are the mean and standard
        deviation of the link heights, respectively; ``R[i,2]`` is the number
        of links included in the calculation; and ``R[i,3]`` is the
        inconsistency coefficient,

        .. math:: \\frac{\\mathtt{Z[i,2]} - \\mathtt{R[i,0]}} {R[i,1]}

    Notes
    -----
    This function behaves similarly to the MATLAB(TM) ``inconsistent``
    function.

    Examples
    --------
    >>> from scipy.cluster.hierarchy import inconsistent, linkage
    >>> from matplotlib import pyplot as plt
    >>> X = [[i] for i in [2, 8, 0, 4, 1, 9, 9, 0]]
    >>> Z = linkage(X, 'ward')
    >>> print(Z)
    [[ 5.          6.          0.          2.        ]
     [ 2.          7.          0.          2.        ]
     [ 0.          4.          1.          2.        ]
     [ 1.          8.          1.15470054  3.        ]
     [ 9.         10.          2.12132034  4.        ]
     [ 3.         12.          4.11096096  5.        ]
     [11.         13.         14.07183949  8.        ]]
    >>> inconsistent(Z)
    array([[ 0.        ,  0.        ,  1.        ,  0.        ],
           [ 0.        ,  0.        ,  1.        ,  0.        ],
           [ 1.        ,  0.        ,  1.        ,  0.        ],
           [ 0.57735027,  0.81649658,  2.        ,  0.70710678],
           [ 1.04044011,  1.06123822,  3.        ,  1.01850858],
           [ 3.11614065,  1.40688837,  2.        ,  0.70710678],
           [ 6.44583366,  6.76770586,  3.        ,  1.12682288]])

    """
    xp = array_namespace(Z)
    Z = as_xparray(Z, order='C', dtype=xp.float64, xp=xp)
    is_valid_linkage(Z, throw=True, name='Z')
    if not d == np.floor(d) or d < 0:
        raise ValueError('The second argument d must be a nonnegative integer value.')
    n = Z.shape[0] + 1
    R = np.zeros((n - 1, 4), dtype=np.float64)
    Z = np.asarray(Z)
    _hierarchy.inconsistent(Z, R, int(n), int(d))
    R = xp.asarray(R)
    return R