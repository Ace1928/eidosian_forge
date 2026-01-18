from __future__ import annotations
from collections import defaultdict
from typing import Literal
import numpy as np
import rustworkx as rx
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.custom_iterator import CustomIterator
from qiskit.quantum_info.operators.mixins import GroupMixin, LinearMixin
from qiskit.quantum_info.operators.symplectic.base_pauli import BasePauli
from qiskit.quantum_info.operators.symplectic.clifford import Clifford
from qiskit.quantum_info.operators.symplectic.pauli import Pauli
@classmethod
def from_symplectic(cls, z: np.ndarray, x: np.ndarray, phase: np.ndarray | None=0) -> PauliList:
    """Construct a PauliList from a symplectic data.

        Args:
            z (np.ndarray): 2D boolean Numpy array.
            x (np.ndarray): 2D boolean Numpy array.
            phase (np.ndarray or None): Optional, 1D integer array from Z_4.

        Returns:
            PauliList: the constructed PauliList.
        """
    base_z, base_x, base_phase = cls._from_array(z, x, phase)
    return cls(BasePauli(base_z, base_x, base_phase))