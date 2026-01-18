import numpy as np
from qiskit.circuit.library.standard_gates import RXGate, RZGate, RYGate
def place_unitary(unitary: np.ndarray, n: int, j: int) -> np.ndarray:
    """
    Computes I(j - 1) tensor product U tensor product I(n - j), where U is a unitary matrix
    of size ``(2, 2)``.

    Args:
        unitary: a unitary matrix of size ``(2, 2)``.
        n: num qubits.
        j: position where to place a unitary.

    Returns:
        a unitary of n qubits with u in position j.
    """
    return np.kron(np.kron(np.eye(2 ** j), unitary), np.eye(2 ** (n - 1 - j)))