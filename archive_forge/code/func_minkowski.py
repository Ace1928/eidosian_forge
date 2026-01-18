import cupy
def minkowski(u, v, p):
    """Compute the Minkowski distance between two 1-D arrays.

    Args:
        u (array_like): Input array of size (N,)
        v (array_like): Input array of size (N,)
        p (float): The order of the norm of the difference
            :math:`{\\|u-v\\|}_p`. Note that for :math:`0 < p < 1`,
            the triangle inequality only holds with an additional
            multiplicative factor, i.e. it is only a quasi-metric.

    Returns:
        minkowski (double): The Minkowski distance between vectors `u` and `v`.
    """
    if not pylibraft_available:
        raise RuntimeError('pylibraft is not installed')
    u = cupy.asarray(u)
    v = cupy.asarray(v)
    output_arr = cupy.zeros((1,), dtype=u.dtype)
    pairwise_distance(u, v, output_arr, 'minkowski', p)
    return output_arr[0]