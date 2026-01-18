import cupy
def correlation(u, v):
    """Compute the correlation distance between two 1-D arrays.

    The correlation distance is defined as

    .. math::
        d(u, v) = 1 - \\frac{(u - \\bar{u}) \\cdot (v - \\bar{v})}{
        \\|(u - \\bar{u})\\|_2 \\|(v - \\bar{v})\\|_2}

    where :math:`\\bar{u}` is the mean of the elements of :math:`u` and
    :math:`x \\cdot y` is the dot product.

    Args:
        u (array_like): Input array of size (N,)
        v (array_like): Input array of size (N,)

    Returns:
        correlation (double): The correlation distance between
        vectors `u` and `v`.
    """
    if not pylibraft_available:
        raise RuntimeError('pylibraft is not installed')
    u = cupy.asarray(u)
    v = cupy.asarray(v)
    output_arr = cupy.zeros((1,), dtype=u.dtype)
    pairwise_distance(u, v, output_arr, 'correlation')
    return output_arr[0]