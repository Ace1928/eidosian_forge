import itertools
import os
import time
from collections import defaultdict
from random import randint, random, sample, randrange
from typing import Iterator, Optional, Tuple, TYPE_CHECKING
import numpy as np
import pytest
import sympy
import cirq
from cirq import circuits
from cirq import ops
from cirq.testing.devices import ValidatingTestDevice
@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_composite_gate_to_unitary_matrix(circuit_cls):

    class CnotComposite(cirq.testing.TwoQubitGate):

        def _decompose_(self, qubits):
            q0, q1 = qubits
            return (cirq.Y(q1) ** (-0.5), cirq.CZ(q0, q1), cirq.Y(q1) ** 0.5)
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = circuit_cls(cirq.X(a), CnotComposite()(a, b), cirq.X(a), cirq.measure(a), cirq.X(b), cirq.measure(b))
    assert cirq.has_unitary(c)
    mat = c.unitary()
    mat_expected = cirq.unitary(cirq.CNOT)
    cirq.testing.assert_allclose_up_to_global_phase(mat, mat_expected, atol=1e-08)