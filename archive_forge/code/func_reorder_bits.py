from __future__ import annotations
from typing import List
from dataclasses import dataclass
from qiskit import circuit
from qiskit.circuit.quantumregister import Qubit, QuantumRegister
from qiskit.transpiler.exceptions import LayoutError
from qiskit.converters import isinstanceint
def reorder_bits(self, bits) -> list[int]:
    """Given an ordered list of bits, reorder them according to this layout.

        The list of bits must exactly match the virtual bits in this layout.

        Args:
            bits (list[Bit]): the bits to reorder.

        Returns:
            List: ordered bits.
        """
    order = [0] * len(bits)
    for i, v in enumerate(bits):
        j = self[v]
        order[i] = j
    return order