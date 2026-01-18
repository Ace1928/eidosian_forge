from typing import Optional, Union
import numpy as np
from qiskit.exceptions import QiskitError
def calc_inverse_matrix(mat: np.ndarray, verify: bool=False):
    """Given a square numpy(dtype=int) matrix mat, tries to compute its inverse.

    Args:
        mat: a boolean square matrix.
        verify: if True asserts that the multiplication of mat and its inverse is the identity matrix.

    Returns:
        np.ndarray: the inverse matrix.

    Raises:
         QiskitError: if the matrix is not square.
         QiskitError: if the matrix is not invertible.
    """
    if mat.shape[0] != mat.shape[1]:
        raise QiskitError('Matrix to invert is a non-square matrix.')
    n = mat.shape[0]
    mat1 = np.concatenate((mat, np.eye(n, dtype=int)), axis=1)
    mat1 = _gauss_elimination(mat1, None, full_elim=True)
    r = _compute_rank_after_gauss_elim(mat1[:, 0:n])
    if r < n:
        raise QiskitError('The matrix is not invertible.')
    matinv = mat1[:, n:2 * n]
    if verify:
        mat2 = np.dot(mat, matinv) % 2
        assert np.array_equal(mat2, np.eye(n))
    return matinv