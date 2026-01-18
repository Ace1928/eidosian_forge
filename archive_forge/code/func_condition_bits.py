from __future__ import annotations
import copy
from itertools import zip_longest
import math
from typing import List, Type
import numpy
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.classicalregister import ClassicalRegister, Clbit
from qiskit.qobj.qasm_qobj import QasmQobjInstruction
from qiskit.circuit.parameter import ParameterExpression
from qiskit.circuit.operation import Operation
from qiskit.circuit.annotated_operation import AnnotatedOperation, InverseModifier
@property
def condition_bits(self) -> List[Clbit]:
    """Get Clbits in condition."""
    from qiskit.circuit.controlflow import condition_resources
    if self.condition is None:
        return []
    return list(condition_resources(self.condition).clbits)