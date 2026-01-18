import numpy as np
import pytest
import cirq
import cirq.testing
def test_bloch_vector_multi_pure():
    plus_plus_state = np.array([0.5, 0.5, 0.5, 0.5])
    bloch_0 = cirq.bloch_vector_from_state_vector(plus_plus_state, 0)
    bloch_1 = cirq.bloch_vector_from_state_vector(plus_plus_state, 1)
    desired_simple = np.array([1, 0, 0])
    np.testing.assert_array_almost_equal(bloch_1, desired_simple)
    np.testing.assert_array_almost_equal(bloch_0, desired_simple)