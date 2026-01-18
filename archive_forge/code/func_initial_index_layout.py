from __future__ import annotations
from typing import List
from dataclasses import dataclass
from qiskit import circuit
from qiskit.circuit.quantumregister import Qubit, QuantumRegister
from qiskit.transpiler.exceptions import LayoutError
from qiskit.converters import isinstanceint
def initial_index_layout(self, filter_ancillas: bool=False) -> List[int]:
    """Generate an initial layout as an array of integers

        Args:
            filter_ancillas: If set to ``True`` any ancilla qubits added
                to the transpiler will not be included in the output.

        Return:
            A layout array that maps a position in the array to its new position in the output
            circuit.
        """
    virtual_map = self.initial_layout.get_virtual_bits()
    if filter_ancillas:
        output = [None] * self._input_qubit_count
    else:
        output = [None] * len(virtual_map)
    for index, (virt, phys) in enumerate(virtual_map.items()):
        if filter_ancillas and index >= self._input_qubit_count:
            break
        pos = self.input_qubit_mapping[virt]
        output[pos] = phys
    return output