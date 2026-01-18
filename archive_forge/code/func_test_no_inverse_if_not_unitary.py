from typing import AbstractSet, Iterator, Any
import pytest
import numpy as np
import sympy
import cirq
def test_no_inverse_if_not_unitary():

    class TestGate(cirq.Gate):

        def _num_qubits_(self):
            return 1

        def _decompose_(self, qubits):
            return cirq.amplitude_damp(0.5).on(qubits[0])
    assert cirq.inverse(TestGate(), None) is None