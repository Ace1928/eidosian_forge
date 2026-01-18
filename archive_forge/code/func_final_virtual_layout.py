from __future__ import annotations
from typing import List
from dataclasses import dataclass
from qiskit import circuit
from qiskit.circuit.quantumregister import Qubit, QuantumRegister
from qiskit.transpiler.exceptions import LayoutError
from qiskit.converters import isinstanceint
def final_virtual_layout(self, filter_ancillas: bool=True) -> Layout:
    """Generate the final layout as a :class:`.Layout` object

        This method will generate an array of final positions for each qubit in the output circuit.
        For example, if you had an input circuit like::

            qc = QuantumCircuit(3)
            qc.h(0)
            qc.cx(0, 1)
            qc.cx(0, 2)

        and the output from the transpiler was::

            tqc = QuantumCircuit(3)
            qc.h(2)
            qc.cx(2, 1)
            qc.swap(0, 1)
            qc.cx(2, 1)

        then the return from this function would be a layout object::

            Layout({
                qc.qubits[0]: 2,
                qc.qubits[1]: 0,
                qc.qubits[2]: 1,
            })

        because qubit 0 in the original circuit's final state is on qubit 3 in the output circuit,
        qubit 1 in the original circuit's final state is on qubit 0, and qubit 2's final state is
        on qubit. The output list length will be as wide as the input circuit's number of qubits,
        as the output list from this method is for tracking the permutation of qubits in the
        original circuit caused by the transpiler.

        Args:
            filter_ancillas: If set to ``False`` any ancillas allocated in the output circuit will be
                included in the layout.

        Returns:
            A layout object mapping to the final positions for each qubit
        """
    res = self.final_index_layout(filter_ancillas=filter_ancillas)
    pos_to_virt = {v: k for k, v in self.input_qubit_mapping.items()}
    return Layout({pos_to_virt[index]: phys for index, phys in enumerate(res)})