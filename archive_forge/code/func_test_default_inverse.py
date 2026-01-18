from typing import AbstractSet, Iterator, Any
import pytest
import numpy as np
import sympy
import cirq
def test_default_inverse():

    class TestGate(cirq.Gate):

        def _num_qubits_(self):
            return 3

        def _decompose_(self, qubits):
            return (cirq.X ** 0.1).on_each(*qubits)
    assert cirq.inverse(TestGate(), None) is not None
    cirq.testing.assert_has_consistent_qid_shape(cirq.inverse(TestGate()))
    cirq.testing.assert_has_consistent_qid_shape(cirq.inverse(TestGate().on(*cirq.LineQubit.range(3))))