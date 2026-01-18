import collections.abc
import pathlib
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_gate_operation_eq():
    g1 = cirq.testing.SingleQubitGate()
    g2 = cirq.testing.SingleQubitGate()
    g3 = cirq.testing.TwoQubitGate()
    r1 = [cirq.NamedQubit('r1')]
    r2 = [cirq.NamedQubit('r2')]
    r12 = r1 + r2
    r21 = r2 + r1
    eq = cirq.testing.EqualsTester()
    eq.make_equality_group(lambda: cirq.GateOperation(g1, r1))
    eq.make_equality_group(lambda: cirq.GateOperation(g2, r1))
    eq.make_equality_group(lambda: cirq.GateOperation(g1, r2))
    eq.make_equality_group(lambda: cirq.GateOperation(g3, r12))
    eq.make_equality_group(lambda: cirq.GateOperation(g3, r21))
    eq.add_equality_group(cirq.GateOperation(cirq.CZ, r21), cirq.GateOperation(cirq.CZ, r12))

    @cirq.value_equality
    class PairGate(cirq.Gate, cirq.InterchangeableQubitsGate):
        """Interchangeable subsets."""

        def __init__(self, num_qubits):
            self._num_qubits = num_qubits

        def num_qubits(self) -> int:
            return self._num_qubits

        def qubit_index_to_equivalence_group_key(self, index: int):
            return index // 2

        def _value_equality_values_(self):
            return (self.num_qubits(),)

    def p(*q):
        return PairGate(len(q)).on(*q)
    a0, a1, b0, b1, c0 = cirq.LineQubit.range(5)
    eq.add_equality_group(p(a0, a1, b0, b1), p(a1, a0, b1, b0))
    eq.add_equality_group(p(b0, b1, a0, a1))
    eq.add_equality_group(p(a0, a1, b0, b1, c0), p(a1, a0, b1, b0, c0))
    eq.add_equality_group(p(a0, b0, a1, b1, c0))
    eq.add_equality_group(p(a0, c0, b0, b1, a1))
    eq.add_equality_group(p(b0, a1, a0, b1, c0))