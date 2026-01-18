import re
import numpy as np
import pytest
import sympy
import cirq
def test_matrix_gate_pow():
    t = sympy.Symbol('t')
    assert cirq.pow(cirq.MatrixGate(1j * np.eye(1)), t, default=None) is None
    assert cirq.pow(cirq.MatrixGate(1j * np.eye(1)), 2) == cirq.MatrixGate(-np.eye(1))
    m = cirq.MatrixGate(np.diag([1, 1j, -1]), qid_shape=(3,))
    assert m ** 3 == cirq.MatrixGate(np.diag([1, -1j, -1]), qid_shape=(3,))