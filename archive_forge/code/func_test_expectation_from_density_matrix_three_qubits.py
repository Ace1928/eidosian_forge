import numpy as np
import pytest
import cirq
def test_expectation_from_density_matrix_three_qubits():
    q0 = cirq.NamedQubit('q0')
    q1 = cirq.NamedQubit('q1')
    q2 = cirq.NamedQubit('q2')
    d_1qbit = cirq.ProjectorString({q1: 1})
    d_2qbits = cirq.ProjectorString({q0: 0, q1: 1})
    state = cirq.testing.random_density_matrix(8)
    np.testing.assert_allclose(d_2qbits.expectation_from_density_matrix(state, {q0: 0, q1: 1, q2: 2}), sum((state[i][i].real for i in [2, 3])))
    np.testing.assert_allclose(d_1qbit.expectation_from_density_matrix(state, {q0: 0, q1: 1, q2: 2}), sum((state[i][i].real for i in [2, 3, 6, 7])))
    np.testing.assert_allclose(d_2qbits.expectation_from_density_matrix(state, {q0: 0, q1: 2, q2: 1}), sum((state[i][i].real for i in [1, 3])))
    np.testing.assert_allclose(d_1qbit.expectation_from_density_matrix(state, {q0: 0, q1: 2, q2: 1}), sum((state[i][i].real for i in [1, 3, 5, 7])))