from typing import Optional
import warnings
import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister, CircuitInstruction
from ..blueprintcircuit import BlueprintCircuit
@approximation_degree.setter
def approximation_degree(self, approximation_degree: int) -> None:
    """Set the approximation degree of the QFT.

        Args:
            approximation_degree: The new approximation degree.

        Raises:
            ValueError: If the approximation degree is smaller than 0.
        """
    if approximation_degree < 0:
        raise ValueError('Approximation degree cannot be smaller than 0.')
    if approximation_degree != self._approximation_degree:
        self._invalidate()
        self._approximation_degree = approximation_degree