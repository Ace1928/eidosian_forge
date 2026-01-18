from __future__ import annotations
import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit, Gate
from qiskit.circuit.exceptions import CircuitError
from qiskit.synthesis.linear import check_invertible_binary_matrix
from qiskit.circuit.library.generalized_gates.permutation import PermutationGate
from qiskit.quantum_info import Clifford
def is_permutation(self) -> bool:
    """Returns whether this linear function is a permutation,
        that is whether every row and every column of the n x n matrix
        has exactly one 1.
        """
    linear = self.linear
    perm = np.all(np.sum(linear, axis=0) == 1) and np.all(np.sum(linear, axis=1) == 1)
    return perm