import cmath
import math
import numbers
from typing import (
import numpy as np
import sympy
import cirq
from cirq import value, protocols, linalg, qis
from cirq._doc import document
from cirq._import import LazyLoader
from cirq.ops import (
from cirq.type_workarounds import NotImplementedType
def pass_operations_over(self, ops: Iterable['cirq.Operation'], after_to_before: bool=False) -> 'PauliString':
    """Determines how the Pauli string changes when conjugated by Cliffords.

        The output and input pauli strings are related by a circuit equivalence.
        In particular, this circuit:

            ───ops───INPUT_PAULI_STRING───

        will be equivalent to this circuit:

            ───OUTPUT_PAULI_STRING───ops───

        up to global phase (assuming `after_to_before` is not set).

        If ops together have matrix C, the Pauli string has matrix P, and the
        output Pauli string has matrix P', then P' == C^-1 P C up to
        global phase.

        Setting `after_to_before` inverts the relationship, so that the output
        is the input and the input is the output. Equivalently, it inverts C.

        Args:
            ops: The operations to move over the string.
            after_to_before: Determines whether the operations start after the
                pauli string, instead of before (and so are moving in the
                opposite direction).
        """
    pauli_map = dict(self._qubit_pauli_map)
    should_negate = False
    for op in ops:
        if pauli_map.keys().isdisjoint(set(op.qubits)):
            continue
        decomposed = _decompose_into_cliffords(op)
        if not after_to_before:
            decomposed = decomposed[::-1]
        for clifford_op in decomposed:
            if pauli_map.keys().isdisjoint(set(clifford_op.qubits)):
                continue
            should_negate ^= _pass_operation_over(pauli_map, clifford_op, after_to_before)
    coef = -self._coefficient if should_negate else self.coefficient
    return PauliString(qubit_pauli_map=pauli_map, coefficient=coef)