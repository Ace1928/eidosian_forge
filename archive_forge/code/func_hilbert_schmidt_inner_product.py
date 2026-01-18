from typing import Dict, Tuple
import numpy as np
from cirq import value
from cirq._doc import document
def hilbert_schmidt_inner_product(m1: np.ndarray, m2: np.ndarray) -> complex:
    """Computes Hilbert-Schmidt inner product of two matrices.

    Linear in second argument.
    """
    return np.einsum('ij,ij', m1.conj(), m2)