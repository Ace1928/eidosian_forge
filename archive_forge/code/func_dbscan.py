import warnings
from numbers import Integral, Real
import numpy as np
from scipy import sparse
from ..base import BaseEstimator, ClusterMixin, _fit_context
from ..metrics.pairwise import _VALID_METRICS
from ..neighbors import NearestNeighbors
from ..utils._param_validation import Interval, StrOptions, validate_params
from ..utils.validation import _check_sample_weight
from ._dbscan_inner import dbscan_inner
@validate_params({'X': ['array-like', 'sparse matrix'], 'sample_weight': ['array-like', None]}, prefer_skip_nested_validation=False)
def dbscan(X, eps=0.5, *, min_samples=5, metric='minkowski', metric_params=None, algorithm='auto', leaf_size=30, p=2, sample_weight=None, n_jobs=None):
    """Perform DBSCAN clustering from vector array or distance matrix.

    Read more in the :ref:`User Guide <dbscan>`.

    Parameters
    ----------
    X : {array-like, sparse (CSR) matrix} of shape (n_samples, n_features) or             (n_samples, n_samples)
        A feature array, or array of distances between samples if
        ``metric='precomputed'``.

    eps : float, default=0.5
        The maximum distance between two samples for one to be considered
        as in the neighborhood of the other. This is not a maximum bound
        on the distances of points within a cluster. This is the most
        important DBSCAN parameter to choose appropriately for your data set
        and distance function.

    min_samples : int, default=5
        The number of samples (or total weight) in a neighborhood for a point
        to be considered as a core point. This includes the point itself.

    metric : str or callable, default='minkowski'
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string or callable, it must be one of
        the options allowed by :func:`sklearn.metrics.pairwise_distances` for
        its metric parameter.
        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square during fit.
        X may be a :term:`sparse graph <sparse graph>`,
        in which case only "nonzero" elements may be considered neighbors.

    metric_params : dict, default=None
        Additional keyword arguments for the metric function.

        .. versionadded:: 0.19

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        The algorithm to be used by the NearestNeighbors module
        to compute pointwise distances and find nearest neighbors.
        See NearestNeighbors module documentation for details.

    leaf_size : int, default=30
        Leaf size passed to BallTree or cKDTree. This can affect the speed
        of the construction and query, as well as the memory required
        to store the tree. The optimal value depends
        on the nature of the problem.

    p : float, default=2
        The power of the Minkowski metric to be used to calculate distance
        between points.

    sample_weight : array-like of shape (n_samples,), default=None
        Weight of each sample, such that a sample with a weight of at least
        ``min_samples`` is by itself a core sample; a sample with negative
        weight may inhibit its eps-neighbor from being core.
        Note that weights are absolute, and default to 1.

    n_jobs : int, default=None
        The number of parallel jobs to run for neighbors search. ``None`` means
        1 unless in a :obj:`joblib.parallel_backend` context. ``-1`` means
        using all processors. See :term:`Glossary <n_jobs>` for more details.
        If precomputed distance are used, parallel execution is not available
        and thus n_jobs will have no effect.

    Returns
    -------
    core_samples : ndarray of shape (n_core_samples,)
        Indices of core samples.

    labels : ndarray of shape (n_samples,)
        Cluster labels for each point.  Noisy samples are given the label -1.

    See Also
    --------
    DBSCAN : An estimator interface for this clustering algorithm.
    OPTICS : A similar estimator interface clustering at multiple values of
        eps. Our implementation is optimized for memory usage.

    Notes
    -----
    For an example, see :ref:`examples/cluster/plot_dbscan.py
    <sphx_glr_auto_examples_cluster_plot_dbscan.py>`.

    This implementation bulk-computes all neighborhood queries, which increases
    the memory complexity to O(n.d) where d is the average number of neighbors,
    while original DBSCAN had memory complexity O(n). It may attract a higher
    memory complexity when querying these nearest neighborhoods, depending
    on the ``algorithm``.

    One way to avoid the query complexity is to pre-compute sparse
    neighborhoods in chunks using
    :func:`NearestNeighbors.radius_neighbors_graph
    <sklearn.neighbors.NearestNeighbors.radius_neighbors_graph>` with
    ``mode='distance'``, then using ``metric='precomputed'`` here.

    Another way to reduce memory and computation time is to remove
    (near-)duplicate points and use ``sample_weight`` instead.

    :class:`~sklearn.cluster.OPTICS` provides a similar clustering with lower
    memory usage.

    References
    ----------
    Ester, M., H. P. Kriegel, J. Sander, and X. Xu, `"A Density-Based
    Algorithm for Discovering Clusters in Large Spatial Databases with Noise"
    <https://www.dbs.ifi.lmu.de/Publikationen/Papers/KDD-96.final.frame.pdf>`_.
    In: Proceedings of the 2nd International Conference on Knowledge Discovery
    and Data Mining, Portland, OR, AAAI Press, pp. 226-231. 1996

    Schubert, E., Sander, J., Ester, M., Kriegel, H. P., & Xu, X. (2017).
    :doi:`"DBSCAN revisited, revisited: why and how you should (still) use DBSCAN."
    <10.1145/3068335>`
    ACM Transactions on Database Systems (TODS), 42(3), 19.

    Examples
    --------
    >>> from sklearn.cluster import dbscan
    >>> X = [[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]]
    >>> core_samples, labels = dbscan(X, eps=3, min_samples=2)
    >>> core_samples
    array([0, 1, 2, 3, 4])
    >>> labels
    array([ 0,  0,  0,  1,  1, -1])
    """
    est = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, metric_params=metric_params, algorithm=algorithm, leaf_size=leaf_size, p=p, n_jobs=n_jobs)
    est.fit(X, sample_weight=sample_weight)
    return (est.core_sample_indices_, est.labels_)