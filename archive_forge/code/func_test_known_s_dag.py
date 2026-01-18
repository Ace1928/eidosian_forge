import random
from typing import Sequence
import numpy as np
import pytest
import sympy
import cirq
def test_known_s_dag():
    actual = cirq.single_qubit_matrix_to_gates(np.array([[1, 0], [0, -1j]]), tolerance=0.01)
    assert cirq.approx_eq(actual, [cirq.Z ** (-0.5)], atol=1e-09)