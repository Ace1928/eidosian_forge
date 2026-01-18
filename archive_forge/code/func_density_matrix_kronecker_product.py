import dataclasses
from typing import Any, List, Optional, Sequence, Tuple, Union
import numpy as np
from cirq import protocols
from cirq.linalg import predicates
def density_matrix_kronecker_product(t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    """Merges two density matrices into a single unified density matrix.

    The resulting matrix's shape will be `(t1.shape/2 + t2.shape/2) * 2`. In
    other words, if t1 has shape [A,B,C,A,B,C] and t2 has shape [X,Y,Z,X,Y,Z],
    the resulting matrix will have shape [A,B,C,X,Y,Z,A,B,C,X,Y,Z].

    Args:
        t1: The first density matrix.
        t2: The second density matrix.
    Returns:
        A density matrix representing the unified state.
    """
    t = state_vector_kronecker_product(t1, t2)
    t1_len = len(t1.shape)
    t1_dim = int(t1_len / 2)
    t2_len = len(t2.shape)
    t2_dim = int(t2_len / 2)
    shape = t1.shape[:t1_dim] + t2.shape[:t2_dim]
    return np.moveaxis(t, range(t1_len, t1_len + t2_dim), range(t1_dim, t1_dim + t2_dim)).reshape(shape * 2)