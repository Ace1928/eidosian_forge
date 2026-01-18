from __future__ import annotations
from typing import List, Optional, Union
import numpy
from qiskit.circuit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Statevector, Operator, DensityMatrix
from .standard_gates import MCXGate
@property
def reflection_qubits(self):
    """Reflection qubits, on which S0 is applied (if S0 is not user-specified)."""
    if self._reflection_qubits is not None:
        return self._reflection_qubits
    num_state_qubits = self.oracle.num_qubits - self.oracle.num_ancillas
    return list(range(num_state_qubits))