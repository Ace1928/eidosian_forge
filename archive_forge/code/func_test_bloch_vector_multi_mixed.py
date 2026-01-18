import numpy as np
import pytest
import cirq
import cirq.testing
def test_bloch_vector_multi_mixed():
    sqrt = np.sqrt(0.5)
    phi_plus = np.array([sqrt, 0.0, 0.0, sqrt])
    bloch_0 = cirq.bloch_vector_from_state_vector(phi_plus, 0)
    bloch_1 = cirq.bloch_vector_from_state_vector(phi_plus, 1)
    zero = np.zeros(3)
    np.testing.assert_array_almost_equal(bloch_0, zero)
    np.testing.assert_array_almost_equal(bloch_1, zero)
    rcnot_state = np.array([0.90612745, -0.07465783j, -0.37533028j, 0.18023996])
    bloch_mixed_0 = cirq.bloch_vector_from_state_vector(rcnot_state, 0)
    bloch_mixed_1 = cirq.bloch_vector_from_state_vector(rcnot_state, 1)
    true_mixed_0 = np.array([0.0, -0.6532815, 0.6532815])
    true_mixed_1 = np.array([0.0, 0.0, 0.9238795])
    np.testing.assert_array_almost_equal(true_mixed_0, bloch_mixed_0)
    np.testing.assert_array_almost_equal(true_mixed_1, bloch_mixed_1)