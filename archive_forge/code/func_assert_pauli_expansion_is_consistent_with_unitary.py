from typing import Any
import numpy as np
from cirq import protocols
from cirq.linalg import operator_spaces
def assert_pauli_expansion_is_consistent_with_unitary(val: Any) -> None:
    """Checks Pauli expansion against unitary matrix."""
    method = getattr(val, '_pauli_expansion_', None)
    if method is None:
        return
    pauli_expansion = protocols.pauli_expansion(val, default=None)
    if pauli_expansion is None:
        return
    unitary = protocols.unitary(val, None)
    if unitary is None:
        return
    num_qubits = protocols.num_qubits(val, default=unitary.shape[0].bit_length() - 1)
    basis = operator_spaces.kron_bases(operator_spaces.PAULI_BASIS, repeat=num_qubits)
    recovered_unitary = operator_spaces.matrix_from_basis_coefficients(pauli_expansion, basis)
    assert np.allclose(unitary, recovered_unitary, rtol=0, atol=1e-12)