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
def soft_compare(self, other: 'Instruction') -> bool:
    """
        Soft comparison between gates. Their names, number of qubits, and classical
        bit numbers must match. The number of parameters must match. Each parameter
        is compared. If one is a ParameterExpression then it is not taken into
        account.

        Args:
            other (instruction): other instruction.

        Returns:
            bool: are self and other equal up to parameter expressions.
        """
    if self.name != other.name or other.num_qubits != other.num_qubits or other.num_clbits != other.num_clbits or (len(self.params) != len(other.params)):
        return False
    for self_param, other_param in zip_longest(self.params, other.params):
        if isinstance(self_param, ParameterExpression) or isinstance(other_param, ParameterExpression):
            continue
        if isinstance(self_param, numpy.ndarray) and isinstance(other_param, numpy.ndarray):
            if numpy.shape(self_param) == numpy.shape(other_param) and numpy.allclose(self_param, other_param, atol=_CUTOFF_PRECISION):
                continue
        else:
            try:
                if numpy.isclose(self_param, other_param, atol=_CUTOFF_PRECISION):
                    continue
            except TypeError:
                pass
        return False
    return True