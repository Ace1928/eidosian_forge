from typing import Optional, Union
import numpy as np
from qiskit.exceptions import QiskitError
def random_invertible_binary_matrix(num_qubits: int, seed: Optional[Union[np.random.Generator, int]]=None):
    """Generates a random invertible n x n binary matrix.

    Args:
        num_qubits: the matrix size.
        seed: a random seed.

    Returns:
        np.ndarray: A random invertible binary matrix of size num_qubits.
    """
    if isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = np.random.default_rng(seed)
    rank = 0
    while rank != num_qubits:
        mat = rng.integers(2, size=(num_qubits, num_qubits))
        rank = _compute_rank(mat)
    return mat