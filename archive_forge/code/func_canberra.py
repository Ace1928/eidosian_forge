import cupy
def canberra(u, v):
    """Compute the Canberra distance between two 1-D arrays.

    The Canberra distance is defined as

    .. math::
        d(u, v) = \\sum_{i} \\frac{| u_i - v_i |}{|u_i| + |v_i|}

    Args:
        u (array_like): Input array of size (N,)
        v (array_like): Input array of size (N,)

    Returns:
        canberra (double): The Canberra distance between vectors `u` and `v`.
    """
    if not pylibraft_available:
        raise RuntimeError('pylibraft is not installed')
    u = cupy.asarray(u)
    v = cupy.asarray(v)
    output_arr = cupy.zeros((1,), dtype=u.dtype)
    pairwise_distance(u, v, output_arr, 'canberra')
    return output_arr[0]