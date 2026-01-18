import warnings
import bisect
from collections import deque
import numpy as np
from . import _hierarchy, _optimal_leaf_ordering
import scipy.spatial.distance as distance
from scipy._lib._array_api import array_namespace, as_xparray, copy
from scipy._lib._disjoint_set import DisjointSet
def cophenet(Z, Y=None):
    """
    Calculate the cophenetic distances between each observation in
    the hierarchical clustering defined by the linkage ``Z``.

    Suppose ``p`` and ``q`` are original observations in
    disjoint clusters ``s`` and ``t``, respectively and
    ``s`` and ``t`` are joined by a direct parent cluster
    ``u``. The cophenetic distance between observations
    ``i`` and ``j`` is simply the distance between
    clusters ``s`` and ``t``.

    Parameters
    ----------
    Z : ndarray
        The hierarchical clustering encoded as an array
        (see `linkage` function).
    Y : ndarray (optional)
        Calculates the cophenetic correlation coefficient ``c`` of a
        hierarchical clustering defined by the linkage matrix `Z`
        of a set of :math:`n` observations in :math:`m`
        dimensions. `Y` is the condensed distance matrix from which
        `Z` was generated.

    Returns
    -------
    c : ndarray
        The cophentic correlation distance (if ``Y`` is passed).
    d : ndarray
        The cophenetic distance matrix in condensed form. The
        :math:`ij` th entry is the cophenetic distance between
        original observations :math:`i` and :math:`j`.

    See Also
    --------
    linkage :
        for a description of what a linkage matrix is.
    scipy.spatial.distance.squareform :
        transforming condensed matrices into square ones.

    Examples
    --------
    >>> from scipy.cluster.hierarchy import single, cophenet
    >>> from scipy.spatial.distance import pdist, squareform

    Given a dataset ``X`` and a linkage matrix ``Z``, the cophenetic distance
    between two points of ``X`` is the distance between the largest two
    distinct clusters that each of the points:

    >>> X = [[0, 0], [0, 1], [1, 0],
    ...      [0, 4], [0, 3], [1, 4],
    ...      [4, 0], [3, 0], [4, 1],
    ...      [4, 4], [3, 4], [4, 3]]

    ``X`` corresponds to this dataset ::

        x x    x x
        x        x

        x        x
        x x    x x

    >>> Z = single(pdist(X))
    >>> Z
    array([[ 0.,  1.,  1.,  2.],
           [ 2., 12.,  1.,  3.],
           [ 3.,  4.,  1.,  2.],
           [ 5., 14.,  1.,  3.],
           [ 6.,  7.,  1.,  2.],
           [ 8., 16.,  1.,  3.],
           [ 9., 10.,  1.,  2.],
           [11., 18.,  1.,  3.],
           [13., 15.,  2.,  6.],
           [17., 20.,  2.,  9.],
           [19., 21.,  2., 12.]])
    >>> cophenet(Z)
    array([1., 1., 2., 2., 2., 2., 2., 2., 2., 2., 2., 1., 2., 2., 2., 2., 2.,
           2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 1., 1., 2., 2.,
           2., 2., 2., 2., 1., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
           1., 1., 2., 2., 2., 1., 2., 2., 2., 2., 2., 2., 1., 1., 1.])

    The output of the `scipy.cluster.hierarchy.cophenet` method is
    represented in condensed form. We can use
    `scipy.spatial.distance.squareform` to see the output as a
    regular matrix (where each element ``ij`` denotes the cophenetic distance
    between each ``i``, ``j`` pair of points in ``X``):

    >>> squareform(cophenet(Z))
    array([[0., 1., 1., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
           [1., 0., 1., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
           [1., 1., 0., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
           [2., 2., 2., 0., 1., 1., 2., 2., 2., 2., 2., 2.],
           [2., 2., 2., 1., 0., 1., 2., 2., 2., 2., 2., 2.],
           [2., 2., 2., 1., 1., 0., 2., 2., 2., 2., 2., 2.],
           [2., 2., 2., 2., 2., 2., 0., 1., 1., 2., 2., 2.],
           [2., 2., 2., 2., 2., 2., 1., 0., 1., 2., 2., 2.],
           [2., 2., 2., 2., 2., 2., 1., 1., 0., 2., 2., 2.],
           [2., 2., 2., 2., 2., 2., 2., 2., 2., 0., 1., 1.],
           [2., 2., 2., 2., 2., 2., 2., 2., 2., 1., 0., 1.],
           [2., 2., 2., 2., 2., 2., 2., 2., 2., 1., 1., 0.]])

    In this example, the cophenetic distance between points on ``X`` that are
    very close (i.e., in the same corner) is 1. For other pairs of points is 2,
    because the points will be located in clusters at different
    corners - thus, the distance between these clusters will be larger.

    """
    xp = array_namespace(Z, Y)
    Z = as_xparray(Z, order='C', dtype=xp.float64, xp=xp)
    is_valid_linkage(Z, throw=True, name='Z')
    n = Z.shape[0] + 1
    zz = np.zeros(n * (n - 1) // 2, dtype=np.float64)
    Z = np.asarray(Z)
    _hierarchy.cophenetic_distances(Z, zz, int(n))
    zz = xp.asarray(zz)
    if Y is None:
        return zz
    Y = as_xparray(Y, order='C', xp=xp)
    distance.is_valid_y(Y, throw=True, name='Y')
    z = xp.mean(zz)
    y = xp.mean(Y)
    Yy = Y - y
    Zz = zz - z
    numerator = Yy * Zz
    denomA = Yy ** 2
    denomB = Zz ** 2
    c = xp.sum(numerator) / xp.sqrt(xp.sum(denomA) * xp.sum(denomB))
    return (c, zz)