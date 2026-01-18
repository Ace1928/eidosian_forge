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
def reverse_ops(self):
    """For a composite instruction, reverse the order of sub-instructions.

        This is done by recursively reversing all sub-instructions.
        It does not invert any gate.

        Returns:
            qiskit.circuit.Instruction: a new instruction with
                sub-instructions reversed.
        """
    if not self._definition or not self.mutable:
        return self.copy()
    reverse_inst = self.copy(name=self.name + '_reverse')
    reversed_definition = self._definition.copy_empty_like()
    for inst in reversed(self._definition):
        reversed_definition.append(inst.operation.reverse_ops(), inst.qubits, inst.clbits)
    reverse_inst.definition = reversed_definition
    return reverse_inst