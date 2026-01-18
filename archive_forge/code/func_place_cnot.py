import numpy as np
from qiskit.circuit.library.standard_gates import RXGate, RZGate, RYGate
def place_cnot(n: int, j: int, k: int) -> np.ndarray:
    """
    Places a CNOT from j to k.

    Args:
        n: number of qubits.
        j: control qubit.
        k: target qubit.

    Returns:
        a unitary of n qubits with CNOT placed at ``j`` and ``k``.
    """
    if j < k:
        unitary = np.kron(np.kron(np.eye(2 ** j), [[1, 0], [0, 0]]), np.eye(2 ** (n - 1 - j))) + np.kron(np.kron(np.kron(np.kron(np.eye(2 ** j), [[0, 0], [0, 1]]), np.eye(2 ** (k - j - 1))), [[0, 1], [1, 0]]), np.eye(2 ** (n - 1 - k)))
    else:
        unitary = np.kron(np.kron(np.eye(2 ** j), [[1, 0], [0, 0]]), np.eye(2 ** (n - 1 - j))) + np.kron(np.kron(np.kron(np.kron(np.eye(2 ** k), [[0, 1], [1, 0]]), np.eye(2 ** (j - k - 1))), [[0, 0], [0, 1]]), np.eye(2 ** (n - 1 - j)))
    return unitary