from typing import Optional, Union
import numpy as np
from qiskit.exceptions import QiskitError
def check_invertible_binary_matrix(mat: np.ndarray):
    """Check that a binary matrix is invertible.

    Args:
        mat: a binary matrix.

    Returns:
        bool: True if mat in invertible and False otherwise.
    """
    if len(mat.shape) != 2 or mat.shape[0] != mat.shape[1]:
        return False
    rank = _compute_rank(mat)
    return rank == mat.shape[0]