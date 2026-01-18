from typing import List, Optional
import numpy as np
from qiskit.circuit import QuantumRegister, AncillaRegister, QuantumCircuit
from ..blueprintcircuit import BlueprintCircuit
@property
def num_control_qubits(self) -> int:
    """The number of additional control qubits required.

        Note that the total number of ancilla qubits can be obtained by calling the
        method ``num_ancilla_qubits``.

        Returns:
            The number of additional control qubits required (0 or 1).
        """
    return int(self.num_sum_qubits > 2)