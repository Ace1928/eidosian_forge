from typing import Dict, Tuple
import numpy as np
from cirq import value
from cirq._doc import document
def expand_matrix_in_orthogonal_basis(m: np.ndarray, basis: Dict[str, np.ndarray]) -> value.LinearDict[str]:
    """Computes coefficients of expansion of m in basis.

    We require that basis be orthogonal w.r.t. the Hilbert-Schmidt inner
    product. We do not require that basis be orthonormal. Note that Pauli
    basis (I, X, Y, Z) is orthogonal, but not orthonormal.
    """
    return value.LinearDict({name: hilbert_schmidt_inner_product(b, m) / hilbert_schmidt_inner_product(b, b) for name, b in basis.items()})