import numpy as np
import pytest
import cirq
import cirq.testing
def test_initial_state():
    with pytest.raises(ValueError, match='Out of range'):
        _ = cirq.StabilizerStateChForm(initial_state=-31, num_qubits=5)
    with pytest.raises(ValueError, match='Out of range'):
        _ = cirq.StabilizerStateChForm(initial_state=32, num_qubits=5)
    state = cirq.StabilizerStateChForm(initial_state=23, num_qubits=5)
    expected_state_vector = np.zeros(32)
    expected_state_vector[23] = 1
    np.testing.assert_allclose(state.state_vector(), expected_state_vector)