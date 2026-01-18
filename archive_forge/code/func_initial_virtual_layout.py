from __future__ import annotations
from typing import List
from dataclasses import dataclass
from qiskit import circuit
from qiskit.circuit.quantumregister import Qubit, QuantumRegister
from qiskit.transpiler.exceptions import LayoutError
from qiskit.converters import isinstanceint
def initial_virtual_layout(self, filter_ancillas: bool=False) -> Layout:
    """Return a :class:`.Layout` object for the initial layout.

        This returns a mapping of virtual :class:`~.circuit.Qubit` objects in the input
        circuit to the physical qubit selected during layout. This is analogous
        to the :attr:`.initial_layout` attribute.

        Args:
            filter_ancillas: If set to ``True`` only qubits in the input circuit
                will be in the returned layout. Any ancilla qubits added to the
                output circuit will be filtered from the returned object.
        Returns:
            A layout object mapping the input circuit's :class:`~.circuit.Qubit`
            objects to the selected physical qubits.
        """
    if not filter_ancillas:
        return self.initial_layout
    return Layout({k: v for k, v in self.initial_layout.get_virtual_bits().items() if self.input_qubit_mapping[k] < self._input_qubit_count})