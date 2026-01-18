from typing import Tuple, Callable, List
import numpy as np
from cirq.linalg import combinators, predicates, tolerance
def bidiagonalize_unitary_with_special_orthogonals(mat: np.ndarray, *, rtol: float=1e-05, atol: float=1e-08, check_preconditions: bool=True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Finds orthogonal matrices L, R such that L @ matrix @ R is diagonal.

    Args:
        mat: A unitary matrix.
        rtol: Relative numeric error threshold.
        atol: Absolute numeric error threshold.
        check_preconditions: If set, verifies that the input is a unitary matrix
            (to the given tolerances). Defaults to set.

    Returns:
        A triplet (L, d, R) such that L @ mat @ R = diag(d). Both L and R will
        be orthogonal matrices with determinant equal to 1.

    Raises:
        ValueError: Matrices don't meet preconditions (e.g. not real).
    """
    if check_preconditions:
        if not predicates.is_unitary(mat, rtol=rtol, atol=atol):
            raise ValueError('matrix must be unitary.')
    left, right = bidiagonalize_real_matrix_pair_with_symmetric_products(np.real(mat), np.imag(mat), rtol=rtol, atol=atol, check_preconditions=check_preconditions)
    if np.linalg.det(left) < 0:
        left[0, :] *= -1
    if np.linalg.det(right) < 0:
        right[:, 0] *= -1
    diag = combinators.dot(left, mat, right)
    return (left, np.diag(diag), right)