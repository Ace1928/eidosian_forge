import itertools
import math
from typing import List
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_conjugated_by_ordering():

    class OrderSensitiveGate(cirq.Gate):

        def num_qubits(self):
            return 2

        def _decompose_(self, qubits):
            return [cirq.Y(qubits[0]) ** (-0.5), cirq.CNOT(*qubits)]
    a, b = cirq.LineQubit.range(2)
    inp = cirq.Z(b)
    out1 = inp.conjugated_by(OrderSensitiveGate().on(a, b))
    out2 = inp.conjugated_by([cirq.H(a), cirq.CNOT(a, b)])
    out3 = inp.conjugated_by(cirq.CNOT(a, b)).conjugated_by(cirq.H(a))
    assert out1 == out2 == out3 == cirq.X(a) * cirq.Z(b)