import cupy
def jensenshannon(u, v):
    """Compute the Jensen-Shannon distance between two 1-D arrays.

    The Jensen-Shannon distance is defined as

    .. math::
        d(u, v) = \\sqrt{\\frac{KL(u \\| m) + KL(v \\| m)}{2}}

    where :math:`KL` is the Kullback-Leibler divergence and :math:`m` is the
    pointwise mean of `u` and `v`.

    Args:
        u (array_like): Input array of size (N,)
        v (array_like): Input array of size (N,)

    Returns:
        jensenshannon (double): The Jensen-Shannon distance between
        vectors `u` and `v`.
    """
    if not pylibraft_available:
        raise RuntimeError('pylibraft is not installed')
    u = cupy.asarray(u)
    v = cupy.asarray(v)
    output_arr = cupy.zeros((1,), dtype=u.dtype)
    pairwise_distance(u, v, output_arr, 'jensenshannon')
    return output_arr[0]