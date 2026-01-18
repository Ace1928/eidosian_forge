import numpy as np
import pytest
import cirq
import cirq.testing
@pytest.mark.parametrize('global_phase', (1, 1j, np.exp(1j)))
def test_bloch_vector_one_state(global_phase):
    one_state = global_phase * np.array([0, 1])
    bloch = cirq.bloch_vector_from_state_vector(one_state, 0)
    desired_simple = np.array([0, 0, -1])
    np.testing.assert_array_almost_equal(bloch, desired_simple)