import collections
from typing import Union
import numpy as np
import pytest
import sympy
import sympy.parsing.sympy_parser as sympy_parser
import cirq
import cirq.testing
def test_expectation_from_state_vector_two_qubit_states():
    q = cirq.LineQubit.range(2)
    q_map = {x: i for i, x in enumerate(q)}
    psum1 = cirq.Z(q[0]) + 3.2 * cirq.Z(q[1])
    psum2 = -1 * cirq.X(q[0]) + 2 * cirq.X(q[1])
    wf1 = np.array([0, 1, 0, 0], dtype=complex)
    for state in [wf1, wf1.reshape((2, 2))]:
        np.testing.assert_allclose(psum1.expectation_from_state_vector(state, qubit_map=q_map), -2.2, atol=1e-07)
        np.testing.assert_allclose(psum2.expectation_from_state_vector(state, qubit_map=q_map), 0, atol=1e-07)
    wf2 = np.array([1, 1, 1, 1], dtype=complex) / 2
    for state in [wf2, wf2.reshape((2, 2))]:
        np.testing.assert_allclose(psum1.expectation_from_state_vector(state, qubit_map=q_map), 0, atol=1e-07)
        np.testing.assert_allclose(psum2.expectation_from_state_vector(state, qubit_map=q_map), 1, atol=1e-07)
    psum3 = cirq.Z(q[0]) + cirq.X(q[1])
    wf3 = np.array([1, 1, 0, 0], dtype=complex) / np.sqrt(2)
    q_map_2 = {q0: 1, q1: 0}
    for state in [wf3, wf3.reshape((2, 2))]:
        np.testing.assert_allclose(psum3.expectation_from_state_vector(state, qubit_map=q_map), 2, atol=1e-07)
        np.testing.assert_allclose(psum3.expectation_from_state_vector(state, qubit_map=q_map_2), 0, atol=1e-07)