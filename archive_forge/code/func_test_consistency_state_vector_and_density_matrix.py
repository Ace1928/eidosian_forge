import numpy as np
import pytest
import cirq
def test_consistency_state_vector_and_density_matrix():
    q0 = cirq.NamedQubit('q0')
    q1 = cirq.NamedQubit('q1')
    q2 = cirq.NamedQubit('q2')
    state_vector = cirq.testing.random_superposition(8)
    state = np.einsum('i,j->ij', state_vector, np.conj(state_vector))
    for proj_qubit in (q0, q1, q2):
        for proj_idx in [0, 1]:
            d = cirq.ProjectorString({proj_qubit: proj_idx})
            np.testing.assert_allclose(d.expectation_from_state_vector(state_vector, {q0: 0, q1: 1, q2: 2}), d.expectation_from_density_matrix(state, {q0: 0, q1: 1, q2: 2}))