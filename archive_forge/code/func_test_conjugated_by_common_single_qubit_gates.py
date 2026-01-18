import itertools
import math
from typing import List
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_conjugated_by_common_single_qubit_gates():
    a, b = cirq.LineQubit.range(2)
    base_single_qubit_gates = [cirq.I, cirq.X, cirq.Y, cirq.Z, cirq.X ** (-0.5), cirq.Y ** (-0.5), cirq.Z ** (-0.5), cirq.X ** 0.5, cirq.Y ** 0.5, cirq.Z ** 0.5, cirq.H]
    single_qubit_gates = [g ** i for i in range(4) for g in base_single_qubit_gates]
    for p in [cirq.X, cirq.Y, cirq.Z]:
        for g in single_qubit_gates:
            assert p.on(a).conjugated_by(g.on(b)) == p.on(a)
            actual = cirq.unitary(p.on(a).conjugated_by(g.on(a)))
            u = cirq.unitary(g)
            expected = np.conj(u.T) @ cirq.unitary(p) @ u
            assert cirq.allclose_up_to_global_phase(actual, expected, atol=1e-08)