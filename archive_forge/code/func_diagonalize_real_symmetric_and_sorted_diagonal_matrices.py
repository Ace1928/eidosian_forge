from typing import Tuple, Callable, List
import numpy as np
from cirq.linalg import combinators, predicates, tolerance
def diagonalize_real_symmetric_and_sorted_diagonal_matrices(symmetric_matrix: np.ndarray, diagonal_matrix: np.ndarray, *, rtol: float=1e-05, atol: float=1e-08, check_preconditions: bool=True) -> np.ndarray:
    """Returns an orthogonal matrix that diagonalizes both given matrices.

    The given matrices must commute.
    Guarantees that the sorted diagonal matrix is not permuted by the
    diagonalization (except for nearly-equal values).

    Args:
        symmetric_matrix: A real symmetric matrix.
        diagonal_matrix: A real diagonal matrix with entries along the diagonal
            sorted into descending order.
        rtol: Relative numeric error threshold.
        atol: Absolute numeric error threshold.
        check_preconditions: If set, verifies that the input matrices commute
            and are respectively symmetric and diagonal descending.

    Returns:
        An orthogonal matrix P such that P.T @ symmetric_matrix @ P is diagonal
        and P.T @ diagonal_matrix @ P = diagonal_matrix (up to tolerance).

    Raises:
        ValueError: Matrices don't meet preconditions (e.g. not symmetric).
    """
    if check_preconditions:
        if np.any(np.imag(symmetric_matrix)) or not predicates.is_hermitian(symmetric_matrix, rtol=rtol, atol=atol):
            raise ValueError('symmetric_matrix must be real symmetric.')
        if not predicates.is_diagonal(diagonal_matrix, atol=atol) or np.any(np.imag(diagonal_matrix)) or np.any(diagonal_matrix[:-1, :-1] < diagonal_matrix[1:, 1:]):
            raise ValueError('diagonal_matrix must be real diagonal descending.')
        if not predicates.matrix_commutes(diagonal_matrix, symmetric_matrix, rtol=rtol, atol=atol):
            raise ValueError('Given matrices must commute.')

    def similar_singular(i, j):
        return np.allclose(diagonal_matrix[i, i], diagonal_matrix[j, j], rtol=rtol)
    ranges = _contiguous_groups(diagonal_matrix.shape[0], similar_singular)
    p = np.zeros(symmetric_matrix.shape, dtype=np.float64)
    for start, end in ranges:
        block = symmetric_matrix[start:end, start:end]
        p[start:end, start:end] = diagonalize_real_symmetric_matrix(block, rtol=rtol, atol=atol, check_preconditions=False)
    return p