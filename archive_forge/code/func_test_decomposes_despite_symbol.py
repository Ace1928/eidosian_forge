import itertools
import pytest
import numpy as np
import sympy
import cirq
def test_decomposes_despite_symbol():
    q0, q1 = (cirq.NamedQubit('q0'), cirq.NamedQubit('q1'))
    gate = cirq.PauliInteractionGate(cirq.Z, False, cirq.X, False, exponent=sympy.Symbol('x'))
    assert cirq.decompose_once_with_qubits(gate, [q0, q1])