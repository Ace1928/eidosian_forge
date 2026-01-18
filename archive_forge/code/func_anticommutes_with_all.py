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
def anticommutes_with_all(self, other: PauliList) -> np.ndarray:
    """Return indexes of rows that commute other.

        If ``other`` is a multi-row Pauli list the returned vector indexes rows
        of the current PauliList that anti-commute with *all* Paulis in other.
        If no rows satisfy the condition the returned array will be empty.

        Args:
            other (PauliList): a single Pauli or multi-row PauliList.

        Returns:
            array: index array of the anti-commuting rows.
        """
    return self._commutes_with_all(other, anti=True)