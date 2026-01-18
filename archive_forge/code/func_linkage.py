import warnings
import bisect
from collections import deque
import numpy as np
from . import _hierarchy, _optimal_leaf_ordering
import scipy.spatial.distance as distance
from scipy._lib._array_api import array_namespace, as_xparray, copy
from scipy._lib._disjoint_set import DisjointSet
def linkage(y, method='single', metric='euclidean', optimal_ordering=False):
    """
    Perform hierarchical/agglomerative clustering.

    The input y may be either a 1-D condensed distance matrix
    or a 2-D array of observation vectors.

    If y is a 1-D condensed distance matrix,
    then y must be a :math:`\\binom{n}{2}` sized
    vector, where n is the number of original observations paired
    in the distance matrix. The behavior of this function is very
    similar to the MATLAB linkage function.

    A :math:`(n-1)` by 4 matrix ``Z`` is returned. At the
    :math:`i`-th iteration, clusters with indices ``Z[i, 0]`` and
    ``Z[i, 1]`` are combined to form cluster :math:`n + i`. A
    cluster with an index less than :math:`n` corresponds to one of
    the :math:`n` original observations. The distance between
    clusters ``Z[i, 0]`` and ``Z[i, 1]`` is given by ``Z[i, 2]``. The
    fourth value ``Z[i, 3]`` represents the number of original
    observations in the newly formed cluster.

    The following linkage methods are used to compute the distance
    :math:`d(s, t)` between two clusters :math:`s` and
    :math:`t`. The algorithm begins with a forest of clusters that
    have yet to be used in the hierarchy being formed. When two
    clusters :math:`s` and :math:`t` from this forest are combined
    into a single cluster :math:`u`, :math:`s` and :math:`t` are
    removed from the forest, and :math:`u` is added to the
    forest. When only one cluster remains in the forest, the algorithm
    stops, and this cluster becomes the root.

    A distance matrix is maintained at each iteration. The ``d[i,j]``
    entry corresponds to the distance between cluster :math:`i` and
    :math:`j` in the original forest.

    At each iteration, the algorithm must update the distance matrix
    to reflect the distance of the newly formed cluster u with the
    remaining clusters in the forest.

    Suppose there are :math:`|u|` original observations
    :math:`u[0], \\ldots, u[|u|-1]` in cluster :math:`u` and
    :math:`|v|` original objects :math:`v[0], \\ldots, v[|v|-1]` in
    cluster :math:`v`. Recall, :math:`s` and :math:`t` are
    combined to form cluster :math:`u`. Let :math:`v` be any
    remaining cluster in the forest that is not :math:`u`.

    The following are methods for calculating the distance between the
    newly formed cluster :math:`u` and each :math:`v`.

      * method='single' assigns

        .. math::
           d(u,v) = \\min(dist(u[i],v[j]))

        for all points :math:`i` in cluster :math:`u` and
        :math:`j` in cluster :math:`v`. This is also known as the
        Nearest Point Algorithm.

      * method='complete' assigns

        .. math::
           d(u, v) = \\max(dist(u[i],v[j]))

        for all points :math:`i` in cluster u and :math:`j` in
        cluster :math:`v`. This is also known by the Farthest Point
        Algorithm or Voor Hees Algorithm.

      * method='average' assigns

        .. math::
           d(u,v) = \\sum_{ij} \\frac{d(u[i], v[j])}
                                   {(|u|*|v|)}

        for all points :math:`i` and :math:`j` where :math:`|u|`
        and :math:`|v|` are the cardinalities of clusters :math:`u`
        and :math:`v`, respectively. This is also called the UPGMA
        algorithm.

      * method='weighted' assigns

        .. math::
           d(u,v) = (dist(s,v) + dist(t,v))/2

        where cluster u was formed with cluster s and t and v
        is a remaining cluster in the forest (also called WPGMA).

      * method='centroid' assigns

        .. math::
           dist(s,t) = ||c_s-c_t||_2

        where :math:`c_s` and :math:`c_t` are the centroids of
        clusters :math:`s` and :math:`t`, respectively. When two
        clusters :math:`s` and :math:`t` are combined into a new
        cluster :math:`u`, the new centroid is computed over all the
        original objects in clusters :math:`s` and :math:`t`. The
        distance then becomes the Euclidean distance between the
        centroid of :math:`u` and the centroid of a remaining cluster
        :math:`v` in the forest. This is also known as the UPGMC
        algorithm.

      * method='median' assigns :math:`d(s,t)` like the ``centroid``
        method. When two clusters :math:`s` and :math:`t` are combined
        into a new cluster :math:`u`, the average of centroids s and t
        give the new centroid :math:`u`. This is also known as the
        WPGMC algorithm.

      * method='ward' uses the Ward variance minimization algorithm.
        The new entry :math:`d(u,v)` is computed as follows,

        .. math::

           d(u,v) = \\sqrt{\\frac{|v|+|s|}
                               {T}d(v,s)^2
                        + \\frac{|v|+|t|}
                               {T}d(v,t)^2
                        - \\frac{|v|}
                               {T}d(s,t)^2}

        where :math:`u` is the newly joined cluster consisting of
        clusters :math:`s` and :math:`t`, :math:`v` is an unused
        cluster in the forest, :math:`T=|v|+|s|+|t|`, and
        :math:`|*|` is the cardinality of its argument. This is also
        known as the incremental algorithm.

    Warning: When the minimum distance pair in the forest is chosen, there
    may be two or more pairs with the same minimum distance. This
    implementation may choose a different minimum than the MATLAB
    version.

    Parameters
    ----------
    y : ndarray
        A condensed distance matrix. A condensed distance matrix
        is a flat array containing the upper triangular of the distance matrix.
        This is the form that ``pdist`` returns. Alternatively, a collection of
        :math:`m` observation vectors in :math:`n` dimensions may be passed as
        an :math:`m` by :math:`n` array. All elements of the condensed distance
        matrix must be finite, i.e., no NaNs or infs.
    method : str, optional
        The linkage algorithm to use. See the ``Linkage Methods`` section below
        for full descriptions.
    metric : str or function, optional
        The distance metric to use in the case that y is a collection of
        observation vectors; ignored otherwise. See the ``pdist``
        function for a list of valid distance metrics. A custom distance
        function can also be used.
    optimal_ordering : bool, optional
        If True, the linkage matrix will be reordered so that the distance
        between successive leaves is minimal. This results in a more intuitive
        tree structure when the data are visualized. defaults to False, because
        this algorithm can be slow, particularly on large datasets [2]_. See
        also the `optimal_leaf_ordering` function.

        .. versionadded:: 1.0.0

    Returns
    -------
    Z : ndarray
        The hierarchical clustering encoded as a linkage matrix.

    Notes
    -----
    1. For method 'single', an optimized algorithm based on minimum spanning
       tree is implemented. It has time complexity :math:`O(n^2)`.
       For methods 'complete', 'average', 'weighted' and 'ward', an algorithm
       called nearest-neighbors chain is implemented. It also has time
       complexity :math:`O(n^2)`.
       For other methods, a naive algorithm is implemented with :math:`O(n^3)`
       time complexity.
       All algorithms use :math:`O(n^2)` memory.
       Refer to [1]_ for details about the algorithms.
    2. Methods 'centroid', 'median', and 'ward' are correctly defined only if
       Euclidean pairwise metric is used. If `y` is passed as precomputed
       pairwise distances, then it is the user's responsibility to assure that
       these distances are in fact Euclidean, otherwise the produced result
       will be incorrect.

    See Also
    --------
    scipy.spatial.distance.pdist : pairwise distance metrics

    References
    ----------
    .. [1] Daniel Mullner, "Modern hierarchical, agglomerative clustering
           algorithms", :arXiv:`1109.2378v1`.
    .. [2] Ziv Bar-Joseph, David K. Gifford, Tommi S. Jaakkola, "Fast optimal
           leaf ordering for hierarchical clustering", 2001. Bioinformatics
           :doi:`10.1093/bioinformatics/17.suppl_1.S22`

    Examples
    --------
    >>> from scipy.cluster.hierarchy import dendrogram, linkage
    >>> from matplotlib import pyplot as plt
    >>> X = [[i] for i in [2, 8, 0, 4, 1, 9, 9, 0]]

    >>> Z = linkage(X, 'ward')
    >>> fig = plt.figure(figsize=(25, 10))
    >>> dn = dendrogram(Z)

    >>> Z = linkage(X, 'single')
    >>> fig = plt.figure(figsize=(25, 10))
    >>> dn = dendrogram(Z)
    >>> plt.show()
    """
    xp = array_namespace(y)
    y = as_xparray(y, order='C', dtype=xp.float64, xp=xp)
    if method not in _LINKAGE_METHODS:
        raise ValueError(f'Invalid method: {method}')
    if method in _EUCLIDEAN_METHODS and metric != 'euclidean' and (y.ndim == 2):
        msg = f'`method={method}` requires the distance metric to be Euclidean'
        raise ValueError(msg)
    if y.ndim == 1:
        distance.is_valid_y(y, throw=True, name='y')
    elif y.ndim == 2:
        if y.shape[0] == y.shape[1] and np.allclose(np.diag(y), 0) and xp.all(y >= 0) and np.allclose(y, y.T):
            warnings.warn('The symmetric non-negative hollow observation matrix looks suspiciously like an uncondensed distance matrix', ClusterWarning, stacklevel=2)
        y = distance.pdist(y, metric)
        y = xp.asarray(y)
    else:
        raise ValueError('`y` must be 1 or 2 dimensional.')
    if not xp.all(xp.isfinite(y)):
        raise ValueError('The condensed distance matrix must contain only finite values.')
    n = int(distance.num_obs_y(y))
    method_code = _LINKAGE_METHODS[method]
    y = np.asarray(y)
    if method == 'single':
        result = _hierarchy.mst_single_linkage(y, n)
    elif method in ['complete', 'average', 'weighted', 'ward']:
        result = _hierarchy.nn_chain(y, n, method_code)
    else:
        result = _hierarchy.fast_linkage(y, n, method_code)
    result = xp.asarray(result)
    if optimal_ordering:
        y = xp.asarray(y)
        return optimal_leaf_ordering(result, y)
    else:
        return result