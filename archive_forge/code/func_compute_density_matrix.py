from typing import Sequence
import numpy as np
import pytest
import cirq
def compute_density_matrix(circuit: cirq.Circuit, qubits: Sequence[cirq.Qid]) -> np.ndarray:
    """Computes density matrix prepared by circuit based on its unitary."""
    u = circuit.unitary(qubit_order=qubits)
    phi = u[:, 0]
    rho = np.outer(phi, np.conjugate(phi))
    return rho