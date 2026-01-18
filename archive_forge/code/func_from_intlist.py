from __future__ import annotations
from typing import List
from dataclasses import dataclass
from qiskit import circuit
from qiskit.circuit.quantumregister import Qubit, QuantumRegister
from qiskit.transpiler.exceptions import LayoutError
from qiskit.converters import isinstanceint
@staticmethod
def from_intlist(int_list, *qregs):
    """Converts a list of integers to a Layout
        mapping virtual qubits (index of the list) to
        physical qubits (the list values).

        Args:
            int_list (list): A list of integers.
            *qregs (QuantumRegisters): The quantum registers to apply
                the layout to.
        Returns:
            Layout: The corresponding Layout object.
        Raises:
            LayoutError: Invalid input layout.
        """
    if not all((isinstanceint(i) for i in int_list)):
        raise LayoutError('Expected a list of ints')
    if len(int_list) != len(set(int_list)):
        raise LayoutError('Duplicate values not permitted; Layout is bijective.')
    num_qubits = sum((reg.size for reg in qregs))
    if len(int_list) != num_qubits:
        raise LayoutError(f'Integer list length ({len(int_list)}) must equal number of qubits in circuit ({num_qubits}): {int_list}.')
    out = Layout()
    main_idx = 0
    for qreg in qregs:
        for idx in range(qreg.size):
            out[qreg[idx]] = int_list[main_idx]
            main_idx += 1
        out.add_register(qreg)
    if main_idx != len(int_list):
        for int_item in int_list[main_idx:]:
            out[int_item] = None
    return out