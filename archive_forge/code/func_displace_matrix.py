import numpy as np
def displace_matrix(K):
    """Remove negative eigenvalues from the given kernel matrix by adding a multiple
    of the identity matrix.

    This method keeps the eigenvectors of the matrix intact.

    Args:
        K (array[float]): Kernel matrix, assumed to be symmetric.

    Returns:
        array[float]: Kernel matrix with eigenvalues offset by adding the identity.

    **Example:**

    Consider a symmetric matrix with both positive and negative eigenvalues:

    .. code-block :: pycon

        >>> K = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 2]])
        >>> np.linalg.eigvalsh(K)
        array([-1.,  1.,  2.])

    We then can shift all eigenvalues of the matrix by adding the identity matrix
    multiplied with the absolute value of the smallest (the most negative, that is)
    eigenvalue:

    .. code-block :: pycon

        >>> K_displaced = qml.kernels.displace_matrix(K)
        >>> np.linalg.eigvalsh(K_displaced)
        array([0.,  2.,  3.])

    If the input matrix does not have negative eigenvalues, ``displace_matrix``
    does not have any effect.
    """
    wmin = np.linalg.eigvalsh(K)[0]
    if wmin < 0:
        return K - np.eye(K.shape[0]) * wmin
    return K