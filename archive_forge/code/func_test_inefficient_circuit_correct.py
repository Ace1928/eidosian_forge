from typing import Optional, Sequence, Type
import pytest
import cirq
import sympy
import numpy as np
def test_inefficient_circuit_correct():
    t = 0.1
    v = 0.11
    a, b = cirq.LineQubit.range(2)
    assert_optimization_not_broken(cirq.Circuit(cirq.H(b), cirq.CNOT(a, b), cirq.H(b), cirq.CNOT(a, b), cirq.CNOT(b, a), cirq.H(a), cirq.CNOT(a, b), cirq.Z(a) ** t, cirq.Z(b) ** (-t), cirq.CNOT(a, b), cirq.H(a), cirq.Z(b) ** v, cirq.CNOT(a, b), cirq.Z(a) ** (-v), cirq.Z(b) ** (-v)))