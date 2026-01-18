from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, SupportsFloat
import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit
@target_matrix.setter
def target_matrix(self, target_matrix: np.ndarray) -> None:
    """
        Args:
            target_matrix: a matrix to approximate in the optimization procedure.
        """
    self._target_matrix = target_matrix