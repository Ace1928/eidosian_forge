from typing import AbstractSet, Iterator, Any
import pytest
import numpy as np
import sympy
import cirq
def test_default_validation_and_inverse():

    class TestGate(cirq.Gate):

        def _num_qubits_(self):
            return 2

        def _decompose_(self, qubits):
            a, b = qubits
            yield cirq.Z(a)
            yield cirq.S(b)
            yield cirq.X(a)

        def __eq__(self, other):
            return isinstance(other, TestGate)

        def __repr__(self):
            return 'TestGate()'
    a, b = cirq.LineQubit.range(2)
    with pytest.raises(ValueError, match='number of qubits'):
        TestGate().on(a)
    t = TestGate().on(a, b)
    i = t ** (-1)
    assert i ** (-1) == t
    assert t ** (-1) == i
    assert cirq.decompose(i) == [cirq.X(a), cirq.S(b) ** (-1), cirq.Z(a)]
    assert [*i._decompose_()] == [cirq.X(a), cirq.S(b) ** (-1), cirq.Z(a)]
    assert [*i.gate._decompose_([a, b])] == [cirq.X(a), cirq.S(b) ** (-1), cirq.Z(a)]
    cirq.testing.assert_allclose_up_to_global_phase(cirq.unitary(i), cirq.unitary(t).conj().T, atol=1e-08)
    cirq.testing.assert_implements_consistent_protocols(i, local_vals={'TestGate': TestGate})