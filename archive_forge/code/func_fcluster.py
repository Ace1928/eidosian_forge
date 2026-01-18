import warnings
import bisect
from collections import deque
import numpy as np
from . import _hierarchy, _optimal_leaf_ordering
import scipy.spatial.distance as distance
from scipy._lib._array_api import array_namespace, as_xparray, copy
from scipy._lib._disjoint_set import DisjointSet
def fcluster(Z, t, criterion='inconsistent', depth=2, R=None, monocrit=None):
    """
    Form flat clusters from the hierarchical clustering defined by
    the given linkage matrix.

    Parameters
    ----------
    Z : ndarray
        The hierarchical clustering encoded with the matrix returned
        by the `linkage` function.
    t : scalar
        For criteria 'inconsistent', 'distance' or 'monocrit',
         this is the threshold to apply when forming flat clusters.
        For 'maxclust' or 'maxclust_monocrit' criteria,
         this would be max number of clusters requested.
    criterion : str, optional
        The criterion to use in forming flat clusters. This can
        be any of the following values:

          ``inconsistent`` :
              If a cluster node and all its
              descendants have an inconsistent value less than or equal
              to `t`, then all its leaf descendants belong to the
              same flat cluster. When no non-singleton cluster meets
              this criterion, every node is assigned to its own
              cluster. (Default)

          ``distance`` :
              Forms flat clusters so that the original
              observations in each flat cluster have no greater a
              cophenetic distance than `t`.

          ``maxclust`` :
              Finds a minimum threshold ``r`` so that
              the cophenetic distance between any two original
              observations in the same flat cluster is no more than
              ``r`` and no more than `t` flat clusters are formed.

          ``monocrit`` :
              Forms a flat cluster from a cluster node c
              with index i when ``monocrit[j] <= t``.

              For example, to threshold on the maximum mean distance
              as computed in the inconsistency matrix R with a
              threshold of 0.8 do::

                  MR = maxRstat(Z, R, 3)
                  fcluster(Z, t=0.8, criterion='monocrit', monocrit=MR)

          ``maxclust_monocrit`` :
              Forms a flat cluster from a
              non-singleton cluster node ``c`` when ``monocrit[i] <=
              r`` for all cluster indices ``i`` below and including
              ``c``. ``r`` is minimized such that no more than ``t``
              flat clusters are formed. monocrit must be
              monotonic. For example, to minimize the threshold t on
              maximum inconsistency values so that no more than 3 flat
              clusters are formed, do::

                  MI = maxinconsts(Z, R)
                  fcluster(Z, t=3, criterion='maxclust_monocrit', monocrit=MI)
    depth : int, optional
        The maximum depth to perform the inconsistency calculation.
        It has no meaning for the other criteria. Default is 2.
    R : ndarray, optional
        The inconsistency matrix to use for the ``'inconsistent'``
        criterion. This matrix is computed if not provided.
    monocrit : ndarray, optional
        An array of length n-1. `monocrit[i]` is the
        statistics upon which non-singleton i is thresholded. The
        monocrit vector must be monotonic, i.e., given a node c with
        index i, for all node indices j corresponding to nodes
        below c, ``monocrit[i] >= monocrit[j]``.

    Returns
    -------
    fcluster : ndarray
        An array of length ``n``. ``T[i]`` is the flat cluster number to
        which original observation ``i`` belongs.

    See Also
    --------
    linkage : for information about hierarchical clustering methods work.

    Examples
    --------
    >>> from scipy.cluster.hierarchy import ward, fcluster
    >>> from scipy.spatial.distance import pdist

    All cluster linkage methods - e.g., `scipy.cluster.hierarchy.ward`
    generate a linkage matrix ``Z`` as their output:

    >>> X = [[0, 0], [0, 1], [1, 0],
    ...      [0, 4], [0, 3], [1, 4],
    ...      [4, 0], [3, 0], [4, 1],
    ...      [4, 4], [3, 4], [4, 3]]

    >>> Z = ward(pdist(X))

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

    This matrix represents a dendrogram, where the first and second elements
    are the two clusters merged at each step, the third element is the
    distance between these clusters, and the fourth element is the size of
    the new cluster - the number of original data points included.

    `scipy.cluster.hierarchy.fcluster` can be used to flatten the
    dendrogram, obtaining as a result an assignation of the original data
    points to single clusters.

    This assignation mostly depends on a distance threshold ``t`` - the maximum
    inter-cluster distance allowed:

    >>> fcluster(Z, t=0.9, criterion='distance')
    array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12], dtype=int32)

    >>> fcluster(Z, t=1.1, criterion='distance')
    array([1, 1, 2, 3, 3, 4, 5, 5, 6, 7, 7, 8], dtype=int32)

    >>> fcluster(Z, t=3, criterion='distance')
    array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4], dtype=int32)

    >>> fcluster(Z, t=9, criterion='distance')
    array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int32)

    In the first case, the threshold ``t`` is too small to allow any two
    samples in the data to form a cluster, so 12 different clusters are
    returned.

    In the second case, the threshold is large enough to allow the first
    4 points to be merged with their nearest neighbors. So, here, only 8
    clusters are returned.

    The third case, with a much higher threshold, allows for up to 8 data
    points to be connected - so 4 clusters are returned here.

    Lastly, the threshold of the fourth case is large enough to allow for
    all data points to be merged together - so a single cluster is returned.

    """
    xp = array_namespace(Z)
    Z = as_xparray(Z, order='C', dtype=xp.float64, xp=xp)
    is_valid_linkage(Z, throw=True, name='Z')
    n = Z.shape[0] + 1
    T = np.zeros((n,), dtype='i')
    if monocrit is not None:
        monocrit = np.asarray(monocrit, order='C', dtype=np.float64)
    Z = np.asarray(Z)
    monocrit = np.asarray(monocrit)
    if criterion == 'inconsistent':
        if R is None:
            R = inconsistent(Z, depth)
        else:
            R = as_xparray(R, order='C', dtype=xp.float64, xp=xp)
            is_valid_im(R, throw=True, name='R')
            R = np.asarray(R)
        _hierarchy.cluster_in(Z, R, T, float(t), int(n))
    elif criterion == 'distance':
        _hierarchy.cluster_dist(Z, T, float(t), int(n))
    elif criterion == 'maxclust':
        _hierarchy.cluster_maxclust_dist(Z, T, int(n), t)
    elif criterion == 'monocrit':
        _hierarchy.cluster_monocrit(Z, monocrit, T, float(t), int(n))
    elif criterion == 'maxclust_monocrit':
        _hierarchy.cluster_maxclust_monocrit(Z, monocrit, T, int(n), int(t))
    else:
        raise ValueError('Invalid cluster formation criterion: %s' % str(criterion))
    return xp.asarray(T)