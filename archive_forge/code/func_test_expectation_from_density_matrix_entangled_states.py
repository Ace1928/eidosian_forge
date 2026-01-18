import itertools
import math
from typing import List
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_expectation_from_density_matrix_entangled_states():
    q0, q1 = _make_qubits(2)
    z0z1_pauli_map = {q0: cirq.Z, q1: cirq.Z}
    z0z1 = cirq.PauliString(z0z1_pauli_map)
    x0x1_pauli_map = {q0: cirq.X, q1: cirq.X}
    x0x1 = cirq.PauliString(x0x1_pauli_map)
    q_map = {q0: 0, q1: 1}
    wf1 = np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2)
    rho1 = np.kron(wf1, wf1).reshape((4, 4))
    for state in [rho1, rho1.reshape((2, 2, 2, 2))]:
        np.testing.assert_allclose(z0z1.expectation_from_density_matrix(state, q_map), -1)
        np.testing.assert_allclose(x0x1.expectation_from_density_matrix(state, q_map), 1)
    wf2 = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    rho2 = np.kron(wf2, wf2).reshape((4, 4))
    for state in [rho2, rho2.reshape((2, 2, 2, 2))]:
        np.testing.assert_allclose(z0z1.expectation_from_density_matrix(state, q_map), 1)
        np.testing.assert_allclose(x0x1.expectation_from_density_matrix(state, q_map), 1)
    wf3 = np.array([1, 1, 1, 1], dtype=complex) / 2
    rho3 = np.kron(wf3, wf3).reshape((4, 4))
    for state in [rho3, rho3.reshape((2, 2, 2, 2))]:
        np.testing.assert_allclose(z0z1.expectation_from_density_matrix(state, q_map), 0)
        np.testing.assert_allclose(x0x1.expectation_from_density_matrix(state, q_map), 1)