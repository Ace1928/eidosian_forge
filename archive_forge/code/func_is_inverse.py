from typing import Optional
import warnings
import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister, CircuitInstruction
from ..blueprintcircuit import BlueprintCircuit
def is_inverse(self) -> bool:
    """Whether the inverse Fourier transform is implemented.

        Returns:
            True, if the inverse Fourier transform is implemented, False otherwise.
        """
    return self._inverse