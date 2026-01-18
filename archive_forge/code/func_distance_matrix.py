import cupy
def distance_matrix(x, y, p=2.0):
    """Compute the distance matrix.

    Returns the matrix of all pair-wise distances.

    Args:
        x (array_like): Matrix of M vectors in K dimensions.
        y (array_like): Matrix of N vectors in K dimensions.
        p (float): Which Minkowski p-norm to use (1 <= p <= infinity).
            Default=2.0
    Returns:
        result (cupy.ndarray): Matrix containing the distance from every
            vector in `x` to every vector in `y`, (size M, N).
    """
    x = cupy.asarray(x)
    m, k = x.shape
    y = cupy.asarray(y)
    n, kk = y.shape
    if k != kk:
        raise ValueError('x contains %d-dimensional vectors but y contains %d-dimensional vectors' % (k, kk))
    return cdist(x, y, metric='minkowski', p=p)