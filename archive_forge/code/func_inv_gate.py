from __future__ import annotations
import itertools
import numpy as np
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.predicates import is_isometry
from .diagonal import Diagonal
from .uc import UCGate
from .mcg_up_to_diagonal import MCGupDiag
def inv_gate(self):
    """Return the adjoint of the unitary."""
    if self._inverse is None:
        iso_circuit = self._gates_to_uncompute()
        self._inverse = iso_circuit.to_instruction()
    return self._inverse