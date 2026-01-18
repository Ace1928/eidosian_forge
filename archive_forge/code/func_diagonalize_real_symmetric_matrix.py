from typing import Tuple, Callable, List
import numpy as np
from cirq.linalg import combinators, predicates, tolerance
def diagonalize_real_symmetric_matrix(matrix: np.ndarray, *, rtol: float=1e-05, atol: float=1e-08, check_preconditions: bool=True) -> np.ndarray:
    """Returns an orthogonal matrix that diagonalizes the given matrix.

    Args:
        matrix: A real symmetric matrix to diagonalize.
        rtol: Relative error tolerance.
        atol: Absolute error tolerance.
        check_preconditions: If set, verifies that the input matrix is real and
            symmetric.

    Returns:
        An orthogonal matrix P such that P.T @ matrix @ P is diagonal.

    Raises:
        ValueError: Matrix isn't real symmetric.
    """
    if check_preconditions and (np.any(np.imag(matrix) != 0) or not predicates.is_hermitian(matrix, rtol=rtol, atol=atol)):
        raise ValueError('Input must be real and symmetric.')
    _, result = np.linalg.eigh(matrix)
    return result