from __future__ import annotations
import numpy as np
from qiskit.circuit import QuantumRegister, AncillaRegister, QuantumCircuit
from qiskit.circuit.exceptions import CircuitError
from .functional_pauli_rotations import FunctionalPauliRotations
from .linear_pauli_rotations import LinearPauliRotations
from .integer_comparator import IntegerComparator
@property
def mapped_offsets(self) -> np.ndarray:
    """The offsets mapped to the internal representation.

        Returns:
            The mapped offsets.
        """
    mapped_offsets = np.zeros_like(self.offsets)
    for i, (offset, slope, point) in enumerate(zip(self.offsets, self.slopes, self.breakpoints)):
        mapped_offsets[i] = offset - slope * point - sum(mapped_offsets[:i])
    return mapped_offsets