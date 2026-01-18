from __future__ import annotations
import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit, Gate
from qiskit.circuit.exceptions import CircuitError
from qiskit.synthesis.linear import check_invertible_binary_matrix
from qiskit.circuit.library.generalized_gates.permutation import PermutationGate
from qiskit.quantum_info import Clifford
def function_str(self):
    """Return string representation of the linear function
        viewed as a linear transformation.
        """
    out = '('
    mat = self.linear
    for row in range(self.num_qubits):
        first_entry = True
        for col in range(self.num_qubits):
            if mat[row, col]:
                if not first_entry:
                    out += ' + '
                out += 'x_' + str(col)
                first_entry = False
        if row != self.num_qubits - 1:
            out += ', '
    out += ')\n'
    return out