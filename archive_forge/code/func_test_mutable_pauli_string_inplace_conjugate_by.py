import itertools
import math
from typing import List
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_mutable_pauli_string_inplace_conjugate_by():
    a, b, c = cirq.LineQubit.range(3)
    p = cirq.MutablePauliString(cirq.X(a))

    class NoOp(cirq.Operation):

        def __init__(self, *qubits):
            self._qubits = qubits

        @property
        def qubits(self):
            return self._qubits

        def with_qubits(self, *new_qubits):
            raise NotImplementedError()

        def _decompose_(self):
            return []
    p2 = p.inplace_after(cirq.global_phase_operation(1j))
    assert p2 is p and p == cirq.X(a)
    p2 = p.inplace_after(NoOp(a, b))
    assert p2 is p and p == cirq.X(a)
    p2 = p.inplace_after(cirq.H(a))
    assert p2 is p and p == cirq.Z(a)
    p2 = p.inplace_before(cirq.H(a))
    assert p2 is p and p == cirq.X(a)
    p2 = p.inplace_after(cirq.S(a))
    assert p2 is p and p == cirq.Y(a)
    p2 = p.inplace_before(cirq.S(a))
    assert p2 is p and p == cirq.X(a)
    p2 = p.inplace_before(cirq.S(a))
    assert p2 is p and p == -cirq.Y(a)
    p2 = p.inplace_after(cirq.S(a))
    assert p2 is p and p == cirq.X(a)
    p2 = p.inplace_after(cirq.S(a) ** (-1))
    assert p2 is p and p == -cirq.Y(a)
    p2 = p.inplace_before(cirq.S(a) ** (-1))
    assert p2 is p and p == cirq.X(a)
    p2 = p.inplace_after(cirq.S(b))
    assert p2 is p and p == cirq.X(a)
    p2 = p.inplace_after(cirq.CZ(a, b))
    assert p2 is p and p == cirq.X(a) * cirq.Z(b)
    p2 = p.inplace_after(cirq.CZ(a, c))
    assert p2 is p and p == cirq.X(a) * cirq.Z(b) * cirq.Z(c)
    p2 = p.inplace_after(cirq.H(b))
    assert p2 is p and p == cirq.X(a) * cirq.X(b) * cirq.Z(c)
    p2 = p.inplace_after(cirq.CNOT(b, c))
    assert p2 is p and p == -cirq.X(a) * cirq.Y(b) * cirq.Y(c)
    p = cirq.MutablePauliString(cirq.X(a))
    p2 = p.inplace_after(cirq.PauliInteractionGate(cirq.Y, True, cirq.Z, False).on(a, b))
    assert p2 is p and p == cirq.X(a) * cirq.Z(b)
    p = cirq.MutablePauliString(cirq.X(a))
    p2 = p.inplace_after(cirq.PauliInteractionGate(cirq.X, False, cirq.Z, True).on(a, b))
    assert p2 is p and p == cirq.X(a)
    p = cirq.MutablePauliString(cirq.X(a))
    p2 = p.inplace_after(cirq.PauliInteractionGate(cirq.Y, False, cirq.Z, True).on(a, b))
    assert p2 is p and p == -cirq.X(a) * cirq.Z(b)
    p = cirq.MutablePauliString(cirq.X(a))
    p2 = p.inplace_after(cirq.PauliInteractionGate(cirq.Z, False, cirq.Y, True).on(a, b))
    assert p2 is p and p == -cirq.X(a) * cirq.Y(b)
    p = cirq.MutablePauliString(cirq.X(a))
    p2 = p.inplace_after(cirq.PauliInteractionGate(cirq.Z, True, cirq.X, False).on(a, b))
    assert p2 is p and p == cirq.X(a) * cirq.X(b)
    p = cirq.MutablePauliString(cirq.X(a))
    p2 = p.inplace_after(cirq.PauliInteractionGate(cirq.Z, True, cirq.Y, False).on(a, b))
    assert p2 is p and p == cirq.X(a) * cirq.Y(b)