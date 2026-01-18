import cupy
def pdist(X, metric='euclidean', *, out=None, **kwargs):
    """Compute distance between observations in n-dimensional space.

    Args:
        X (array_like): An :math:`m` by :math:`n` array of :math:`m`
            original observations in an :math:`n`-dimensional space.
            Inputs are converted to float type.
        metric (str, optional): The distance metric to use.
            The distance function can be 'canberra', 'chebyshev',
            'cityblock', 'correlation', 'cosine', 'euclidean', 'hamming',
            'hellinger', 'jensenshannon', 'kl_divergence', 'matching',
            'minkowski', 'russellrao', 'sqeuclidean'.
        out (cupy.ndarray, optional): The output array. If not None, the
            distance matrix Y is stored in this array.
        **kwargs (dict, optional): Extra arguments to `metric`: refer to each
            metric documentation for a list of all possible arguments.
            Some possible arguments:
            p (float): The p-norm to apply for Minkowski, weighted and
            unweighted. Default: 2.0

    Returns:
        Y (cupy.ndarray):
            A :math:`m` by :math:`m` distance matrix is
            returned. For each :math:`i` and :math:`j`, the metric
            ``dist(u=X[i], v=X[j])`` is computed and stored in the
            :math:`ij` th entry.
    """
    all_dist = cdist(X, X, metric=metric, out=out, **kwargs)
    up_idx = cupy.triu_indices_from(all_dist, 1)
    return all_dist[up_idx]