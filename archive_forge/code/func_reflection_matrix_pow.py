import dataclasses
from typing import Any, List, Optional, Sequence, Tuple, Union
import numpy as np
from cirq import protocols
from cirq.linalg import predicates
def reflection_matrix_pow(reflection_matrix: np.ndarray, exponent: float):
    """Raises a matrix with two opposing eigenvalues to a power.

    Args:
        reflection_matrix: The matrix to raise to a power.
        exponent: The power to raise the matrix to.

    Returns:
        The given matrix raised to the given power.
    """
    squared_phase = np.dot(reflection_matrix[:, 0], reflection_matrix[0, :])
    phase = complex(np.sqrt(squared_phase))
    i = np.eye(reflection_matrix.shape[0]) * phase
    pos_part = (i + reflection_matrix) * 0.5
    neg_part = (i - reflection_matrix) * 0.5
    pos_factor = phase ** (exponent - 1)
    neg_factor = pos_factor * complex(-1) ** exponent
    pos_part_raised = pos_factor * pos_part
    neg_part_raised = neg_part * neg_factor
    return pos_part_raised + neg_part_raised