from __future__ import annotations
import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit, Gate
from qiskit.circuit.exceptions import CircuitError
from qiskit.synthesis.linear import check_invertible_binary_matrix
from qiskit.circuit.library.generalized_gates.permutation import PermutationGate
from qiskit.quantum_info import Clifford
def extend_with_identity(self, num_qubits: int, positions: list[int]) -> LinearFunction:
    """Extend linear function to a linear function over nq qubits,
        with identities on other subsystems.

        Args:
            num_qubits: number of qubits of the extended function.

            positions: describes the positions of original qubits in the extended
                function's qubits.

        Returns:
            LinearFunction: extended linear function.
        """
    extended_mat = np.eye(num_qubits, dtype=bool)
    for i, pos in enumerate(positions):
        extended_mat[positions, pos] = self.linear[:, i]
    return LinearFunction(extended_mat)