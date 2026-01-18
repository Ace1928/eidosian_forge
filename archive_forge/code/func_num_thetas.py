from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, SupportsFloat
import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit
@property
@abstractmethod
def num_thetas(self) -> int:
    """

        Returns:
            the number of parameters in this optimization problem.
        """
    raise NotImplementedError