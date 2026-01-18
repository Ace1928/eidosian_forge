from __future__ import annotations
from typing import List
from dataclasses import dataclass
from qiskit import circuit
from qiskit.circuit.quantumregister import Qubit, QuantumRegister
from qiskit.transpiler.exceptions import LayoutError
from qiskit.converters import isinstanceint
@staticmethod
def order_based_on_type(value1, value2):
    """decides which one is physical/virtual based on the type. Returns (virtual, physical)"""
    if isinstanceint(value1) and isinstance(value2, (Qubit, type(None))):
        physical = int(value1)
        virtual = value2
    elif isinstanceint(value2) and isinstance(value1, (Qubit, type(None))):
        physical = int(value2)
        virtual = value1
    else:
        raise LayoutError('The map (%s -> %s) has to be a (Bit -> integer) or the other way around.' % (type(value1), type(value2)))
    return (virtual, physical)