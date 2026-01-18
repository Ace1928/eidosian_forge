import itertools
import pytest
import numpy as np
import sympy
import cirq
def test_manual_default_decompose():
    q0, q1, q2 = _make_qubits(3)
    mat = cirq.Circuit(cirq.PauliStringPhasor(cirq.PauliString({q0: cirq.Z})) ** 0.25, cirq.Z(q0) ** (-0.25)).unitary()
    cirq.testing.assert_allclose_up_to_global_phase(mat, np.eye(2), rtol=1e-07, atol=1e-07)
    mat = cirq.Circuit(cirq.PauliStringPhasor(cirq.PauliString({q0: cirq.Y})) ** 0.25, cirq.Y(q0) ** (-0.25)).unitary()
    cirq.testing.assert_allclose_up_to_global_phase(mat, np.eye(2), rtol=1e-07, atol=1e-07)
    mat = cirq.Circuit(cirq.PauliStringPhasor(cirq.PauliString({q0: cirq.Z, q1: cirq.Z, q2: cirq.Z}))).unitary()
    cirq.testing.assert_allclose_up_to_global_phase(mat, np.diag([1, -1, -1, 1, -1, 1, 1, -1]), rtol=1e-07, atol=1e-07)
    mat = cirq.Circuit(cirq.PauliStringPhasor(cirq.PauliString({q0: cirq.Z, q1: cirq.Y, q2: cirq.X})) ** 0.5).unitary()
    cirq.testing.assert_allclose_up_to_global_phase(mat, np.array([[1, 0, 0, -1, 0, 0, 0, 0], [0, 1, -1, 0, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0, 0, 0], [1, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 1], [0, 0, 0, 0, 0, 1, 1, 0], [0, 0, 0, 0, 0, -1, 1, 0], [0, 0, 0, 0, -1, 0, 0, 1]]) / np.sqrt(2), rtol=1e-07, atol=1e-07)