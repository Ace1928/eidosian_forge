from __future__ import annotations
import typing
from collections.abc import Callable, Mapping, Sequence
from itertools import combinations
import numpy
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit import Instruction, Parameter, ParameterVector, ParameterExpression
from qiskit.exceptions import QiskitError
from ..blueprintcircuit import BlueprintCircuit
def get_unentangled_qubits(self) -> set[int]:
    """Get the indices of unentangled qubits in a set.

        Returns:
            The unentangled qubits.
        """
    entangled_qubits = set()
    for i in range(self._reps):
        for j, block in enumerate(self.entanglement_blocks):
            entangler_map = self.get_entangler_map(i, j, block.num_qubits)
            entangled_qubits.update([idx for indices in entangler_map for idx in indices])
    unentangled_qubits = set(range(self.num_qubits)) - entangled_qubits
    return unentangled_qubits