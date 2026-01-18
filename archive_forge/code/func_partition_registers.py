from __future__ import annotations
import dataclasses
from typing import Iterable, Tuple, Set, Union, TypeVar, TYPE_CHECKING
from qiskit.circuit.classical import expr, types
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.register import Register
from qiskit.circuit.classicalregister import ClassicalRegister, Clbit
from qiskit.circuit.quantumregister import QuantumRegister
def partition_registers(registers: Iterable[Register]) -> Tuple[Set[QuantumRegister], Set[ClassicalRegister]]:
    """Partition a sequence of registers into its quantum and classical registers."""
    qregs = set()
    cregs = set()
    for register in registers:
        if isinstance(register, QuantumRegister):
            qregs.add(register)
        elif isinstance(register, ClassicalRegister):
            cregs.add(register)
        else:
            raise CircuitError(f'Unknown register: {register}.')
    return (qregs, cregs)