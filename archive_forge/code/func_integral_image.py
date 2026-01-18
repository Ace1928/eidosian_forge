import numpy as np
def integral_image(image, *, dtype=None):
    """Integral image / summed area table.

    The integral image contains the sum of all elements above and to the
    left of it, i.e.:

    .. math::

       S[m, n] = \\sum_{i \\leq m} \\sum_{j \\leq n} X[i, j]

    Parameters
    ----------
    image : ndarray
        Input image.

    Returns
    -------
    S : ndarray
        Integral image/summed area table of same shape as input image.

    Notes
    -----
    For better accuracy and to avoid potential overflow, the data type of the
    output may differ from the input's when the default dtype of None is used.
    For inputs with integer dtype, the behavior matches that for
    :func:`numpy.cumsum`. Floating point inputs will be promoted to at least
    double precision. The user can set `dtype` to override this behavior.

    References
    ----------
    .. [1] F.C. Crow, "Summed-area tables for texture mapping,"
           ACM SIGGRAPH Computer Graphics, vol. 18, 1984, pp. 207-212.

    """
    if dtype is None and image.real.dtype.kind == 'f':
        dtype = np.promote_types(image.dtype, np.float64)
    S = image
    for i in range(image.ndim):
        S = S.cumsum(axis=i, dtype=dtype)
    return S