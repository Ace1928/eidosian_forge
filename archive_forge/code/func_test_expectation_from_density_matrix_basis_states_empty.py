import numpy as np
import pytest
import cirq
def test_expectation_from_density_matrix_basis_states_empty():
    q0 = cirq.NamedQubit('q0')
    d = cirq.ProjectorString({})
    np.testing.assert_allclose(d.expectation_from_density_matrix(np.array([[1.0, 0.0], [0.0, 0.0]]), {q0: 0}), 1.0)