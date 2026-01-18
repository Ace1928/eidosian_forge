from typing import Tuple, Callable, List
import numpy as np
from cirq.linalg import combinators, predicates, tolerance
def bidiagonalize_real_matrix_pair_with_symmetric_products(mat1: np.ndarray, mat2: np.ndarray, *, rtol: float=1e-05, atol: float=1e-08, check_preconditions: bool=True) -> Tuple[np.ndarray, np.ndarray]:
    """Finds orthogonal matrices that diagonalize both mat1 and mat2.

    Requires mat1 and mat2 to be real.
    Requires mat1.T @ mat2 to be symmetric.
    Requires mat1 @ mat2.T to be symmetric.

    Args:
        mat1: One of the real matrices.
        mat2: The other real matrix.
        rtol: Relative numeric error threshold.
        atol: Absolute numeric error threshold.
        check_preconditions: If set, verifies that the inputs are real, and that
            mat1.T @ mat2 and mat1 @ mat2.T are both symmetric. Defaults to set.

    Returns:
        A tuple (L, R) of two orthogonal matrices, such that both L @ mat1 @ R
        and L @ mat2 @ R are diagonal matrices.

    Raises:
        ValueError: Matrices don't meet preconditions (e.g. not real).
    """
    if check_preconditions:
        if np.any(np.imag(mat1) != 0):
            raise ValueError('mat1 must be real.')
        if np.any(np.imag(mat2) != 0):
            raise ValueError('mat2 must be real.')
        if not predicates.is_hermitian(np.dot(mat1, mat2.T), rtol=rtol, atol=atol):
            raise ValueError('mat1 @ mat2.T must be symmetric.')
        if not predicates.is_hermitian(np.dot(mat1.T, mat2), rtol=rtol, atol=atol):
            raise ValueError('mat1.T @ mat2 must be symmetric.')
    base_left, base_diag, base_right = _svd_handling_empty(np.real(mat1))
    base_diag = np.diag(base_diag)
    dim = base_diag.shape[0]
    rank = dim
    while rank > 0 and tolerance.all_near_zero(base_diag[rank - 1, rank - 1], atol=atol):
        rank -= 1
    base_diag = base_diag[:rank, :rank]
    semi_corrected = combinators.dot(base_left.T, np.real(mat2), base_right.T)
    overlap = semi_corrected[:rank, :rank]
    overlap_adjust = diagonalize_real_symmetric_and_sorted_diagonal_matrices(overlap, base_diag, rtol=rtol, atol=atol, check_preconditions=check_preconditions)
    extra = semi_corrected[rank:, rank:]
    extra_left_adjust, _, extra_right_adjust = _svd_handling_empty(extra)
    left_adjust = combinators.block_diag(overlap_adjust, extra_left_adjust)
    right_adjust = combinators.block_diag(overlap_adjust.T, extra_right_adjust)
    left = np.dot(left_adjust.T, base_left.T)
    right = np.dot(base_right.T, right_adjust.T)
    return (left, right)