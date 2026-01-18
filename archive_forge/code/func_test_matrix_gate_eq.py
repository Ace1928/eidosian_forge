import re
import numpy as np
import pytest
import sympy
import cirq
def test_matrix_gate_eq():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(cirq.MatrixGate(np.eye(1)))
    eq.add_equality_group(cirq.MatrixGate(-np.eye(1)))
    eq.add_equality_group(cirq.MatrixGate(np.diag([1, 1, 1, 1, 1, -1]), qid_shape=(2, 3)))
    eq.add_equality_group(cirq.MatrixGate(np.diag([1, 1, 1, 1, 1, -1]), qid_shape=(3, 2)))