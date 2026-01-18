from __future__ import annotations
import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit, Gate
from qiskit.circuit.exceptions import CircuitError
from qiskit.synthesis.linear import check_invertible_binary_matrix
from qiskit.circuit.library.generalized_gates.permutation import PermutationGate
from qiskit.quantum_info import Clifford
@property
def original_circuit(self):
    """Returns the original circuit used to construct this linear function
        (including None, when the linear function is not constructed from a circuit).
        """
    return self.params[1]