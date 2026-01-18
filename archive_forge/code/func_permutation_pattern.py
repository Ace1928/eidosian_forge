from __future__ import annotations
import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit, Gate
from qiskit.circuit.exceptions import CircuitError
from qiskit.synthesis.linear import check_invertible_binary_matrix
from qiskit.circuit.library.generalized_gates.permutation import PermutationGate
from qiskit.quantum_info import Clifford
def permutation_pattern(self):
    """This method first checks if a linear function is a permutation and raises a
        `qiskit.circuit.exceptions.CircuitError` if not. In the case that this linear function
        is a permutation, returns the permutation pattern.
        """
    if not self.is_permutation():
        raise CircuitError('The linear function is not a permutation')
    linear = self.linear
    locs = np.where(linear == 1)
    return locs[1]