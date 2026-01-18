import functools
from numbers import Integral
import numpy as np
from scipy.sparse import issparse
from ...preprocessing import LabelEncoder
from ...utils import _safe_indexing, check_random_state, check_X_y
from ...utils._param_validation import (
from ..pairwise import _VALID_METRICS, pairwise_distances, pairwise_distances_chunked
@validate_params({'X': ['array-like'], 'labels': ['array-like']}, prefer_skip_nested_validation=True)
def davies_bouldin_score(X, labels):
    """Compute the Davies-Bouldin score.

    The score is defined as the average similarity measure of each cluster with
    its most similar cluster, where similarity is the ratio of within-cluster
    distances to between-cluster distances. Thus, clusters which are farther
    apart and less dispersed will result in a better score.

    The minimum score is zero, with lower values indicating better clustering.

    Read more in the :ref:`User Guide <davies-bouldin_index>`.

    .. versionadded:: 0.20

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        A list of ``n_features``-dimensional data points. Each row corresponds
        to a single data point.

    labels : array-like of shape (n_samples,)
        Predicted labels for each sample.

    Returns
    -------
    score: float
        The resulting Davies-Bouldin score.

    References
    ----------
    .. [1] Davies, David L.; Bouldin, Donald W. (1979).
       `"A Cluster Separation Measure"
       <https://ieeexplore.ieee.org/document/4766909>`__.
       IEEE Transactions on Pattern Analysis and Machine Intelligence.
       PAMI-1 (2): 224-227

    Examples
    --------
    >>> from sklearn.metrics import davies_bouldin_score
    >>> X = [[0, 1], [1, 1], [3, 4]]
    >>> labels = [0, 0, 1]
    >>> davies_bouldin_score(X, labels)
    0.12...
    """
    X, labels = check_X_y(X, labels)
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    n_samples, _ = X.shape
    n_labels = len(le.classes_)
    check_number_of_labels(n_labels, n_samples)
    intra_dists = np.zeros(n_labels)
    centroids = np.zeros((n_labels, len(X[0])), dtype=float)
    for k in range(n_labels):
        cluster_k = _safe_indexing(X, labels == k)
        centroid = cluster_k.mean(axis=0)
        centroids[k] = centroid
        intra_dists[k] = np.average(pairwise_distances(cluster_k, [centroid]))
    centroid_distances = pairwise_distances(centroids)
    if np.allclose(intra_dists, 0) or np.allclose(centroid_distances, 0):
        return 0.0
    centroid_distances[centroid_distances == 0] = np.inf
    combined_intra_dists = intra_dists[:, None] + intra_dists
    scores = np.max(combined_intra_dists / centroid_distances, axis=1)
    return np.mean(scores)