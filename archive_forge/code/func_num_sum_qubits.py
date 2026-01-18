from typing import List, Optional
import numpy as np
from qiskit.circuit import QuantumRegister, AncillaRegister, QuantumCircuit
from ..blueprintcircuit import BlueprintCircuit
@property
def num_sum_qubits(self) -> int:
    """The number of sum qubits in the circuit.

        Returns:
            The number of qubits needed to represent the weighted sum of the qubits.
        """
    if sum(self.weights) > 0:
        return int(np.floor(np.log2(sum(self.weights))) + 1)
    return 1