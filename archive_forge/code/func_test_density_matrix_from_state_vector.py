import numpy as np
import pytest
import cirq
import cirq.testing
def test_density_matrix_from_state_vector():
    test_state = np.array([0.0 - 0.35355339j, 0.0 + 0.35355339j, 0.0 - 0.35355339j, 0.0 + 0.35355339j, 0.0 + 0.35355339j, 0.0 - 0.35355339j, 0.0 + 0.35355339j, 0.0 - 0.35355339j])
    full_rho = cirq.density_matrix_from_state_vector(test_state)
    np.testing.assert_array_almost_equal(full_rho, np.outer(test_state, np.conj(test_state)))
    rho_one = cirq.density_matrix_from_state_vector(test_state, [1])
    true_one = np.array([[0.5 + 0j, 0.5 + 0j], [0.5 + 0j, 0.5 + 0j]])
    np.testing.assert_array_almost_equal(rho_one, true_one)
    rho_two_zero = cirq.density_matrix_from_state_vector(test_state, [0, 2])
    true_two_zero = np.array([[0.25 + 0j, -0.25 + 0j, -0.25 + 0j, 0.25 + 0j], [-0.25 + 0j, 0.25 + 0j, 0.25 + 0j, -0.25 + 0j], [-0.25 + 0j, 0.25 + 0j, 0.25 + 0j, -0.25 + 0j], [0.25 + 0j, -0.25 + 0j, -0.25 + 0j, 0.25 + 0j]])
    np.testing.assert_array_almost_equal(rho_two_zero, true_two_zero)
    rho_two = cirq.density_matrix_from_state_vector(test_state, [2])
    true_two = np.array([[0.5 + 0j, -0.5 + 0j], [-0.5 + 0j, 0.5 + 0j]])
    np.testing.assert_array_almost_equal(rho_two, true_two)
    rho_zero = cirq.density_matrix_from_state_vector(test_state, [0])
    np.testing.assert_array_almost_equal(rho_zero, true_two)