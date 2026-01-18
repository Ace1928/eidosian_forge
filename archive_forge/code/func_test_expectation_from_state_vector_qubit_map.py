import itertools
import math
from typing import List
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_expectation_from_state_vector_qubit_map():
    q0, q1, q2 = _make_qubits(3)
    z = cirq.PauliString({q0: cirq.Z})
    wf = np.array([0, 1, 0, 1, 0, 0, 0, 0], dtype=complex) / np.sqrt(2)
    for state in [wf, wf.reshape((2, 2, 2))]:
        np.testing.assert_allclose(z.expectation_from_state_vector(state, {q0: 0, q1: 1, q2: 2}), 1, atol=1e-08)
        np.testing.assert_allclose(z.expectation_from_state_vector(state, {q0: 0, q1: 2, q2: 1}), 1, atol=1e-08)
        np.testing.assert_allclose(z.expectation_from_state_vector(state, {q0: 1, q1: 0, q2: 2}), 0, atol=1e-08)
        np.testing.assert_allclose(z.expectation_from_state_vector(state, {q0: 1, q1: 2, q2: 0}), 0, atol=1e-09)
        np.testing.assert_allclose(z.expectation_from_state_vector(state, {q0: 2, q1: 0, q2: 1}), -1, atol=1e-08)
        np.testing.assert_allclose(z.expectation_from_state_vector(state, {q0: 2, q1: 1, q2: 0}), -1, atol=1e-08)