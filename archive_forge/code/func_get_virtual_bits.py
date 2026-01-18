from __future__ import annotations
from typing import List
from dataclasses import dataclass
from qiskit import circuit
from qiskit.circuit.quantumregister import Qubit, QuantumRegister
from qiskit.transpiler.exceptions import LayoutError
from qiskit.converters import isinstanceint
def get_virtual_bits(self):
    """
        Returns the dictionary where the keys are virtual (qu)bits and the
        values are physical (qu)bits.
        """
    return self._v2p