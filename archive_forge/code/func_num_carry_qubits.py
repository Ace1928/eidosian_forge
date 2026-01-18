from typing import List, Optional
import numpy as np
from qiskit.circuit import QuantumRegister, AncillaRegister, QuantumCircuit
from ..blueprintcircuit import BlueprintCircuit
@property
def num_carry_qubits(self) -> int:
    """The number of carry qubits required to compute the sum.

        Note that this is not necessarily equal to the number of ancilla qubits, these can
        be queried using ``num_ancilla_qubits``.

        Returns:
            The number of carry qubits required to compute the sum.
        """
    return self.num_sum_qubits - 1