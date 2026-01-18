import numpy as np
def flip_matrix(K):
    """Remove negative eigenvalues from the given kernel matrix by taking the absolute value.

    This method keeps the eigenvectors of the matrix intact.

    Args:
        K (array[float]): Kernel matrix, assumed to be symmetric.

    Returns:
        array[float]: Kernel matrix with flipped negative eigenvalues.

    Reference:
        This method is introduced in
        `Wang, Du, Luo & Tao (2021) <https://doi.org/10.22331/q-2021-08-30-531>`_.

    **Example:**

    Consider a symmetric matrix with both positive and negative eigenvalues:

    .. code-block :: pycon

        >>> K = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 2]])
        >>> np.linalg.eigvalsh(K)
        array([-1.,  1.,  2.])

    We then can invert the sign of all negative eigenvalues of the matrix, obtaining
    non-negative eigenvalues only:

    .. code-block :: pycon

        >>> K_flipped = qml.kernels.flip_matrix(K)
        >>> np.linalg.eigvalsh(K_flipped)
        array([1.,  1.,  2.])

    If the input matrix does not have negative eigenvalues, ``flip_matrix``
    does not have any effect.
    """
    w, v = np.linalg.eigh(K)
    if w[0] < 0:
        w_abs = np.abs(w)
        return v * w_abs @ np.transpose(v)
    return K