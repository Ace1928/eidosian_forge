import logging
from typing import Optional, List, Tuple, Dict
import numpy as np
from qiskit.exceptions import QiskitError
from ..utils import marginal_counts
from ..counts import Counts
def counts_to_vector(counts: Counts, num_qubits: int) -> Tuple[np.ndarray, int]:
    """Transforms Counts to a probability vector"""
    vec = np.zeros(2 ** num_qubits, dtype=float)
    shots = 0
    for key, val in counts.items():
        shots += val
        vec[int(key, 2)] = val
    vec /= shots
    return (vec, shots)