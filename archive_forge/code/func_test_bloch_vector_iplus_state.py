import numpy as np
import pytest
import cirq
import cirq.testing
@pytest.mark.parametrize('global_phase', (1, 1j, np.exp(1j)))
def test_bloch_vector_iplus_state(global_phase):
    sqrt = np.sqrt(0.5)
    iplus_state = global_phase * np.array([sqrt, 1j * sqrt])
    bloch = cirq.bloch_vector_from_state_vector(iplus_state, 0)
    desired_simple = np.array([0, 1, 0])
    np.testing.assert_array_almost_equal(bloch, desired_simple)