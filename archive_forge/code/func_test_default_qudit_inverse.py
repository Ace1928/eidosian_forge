from typing import AbstractSet, Iterator, Any
import pytest
import numpy as np
import sympy
import cirq
def test_default_qudit_inverse():

    class TestGate(cirq.Gate):

        def _qid_shape_(self):
            return (1, 2, 3)

        def _decompose_(self, qubits):
            return (cirq.X ** 0.1).on(qubits[1])
    assert cirq.qid_shape(cirq.inverse(TestGate(), None)) == (1, 2, 3)
    cirq.testing.assert_has_consistent_qid_shape(cirq.inverse(TestGate()))