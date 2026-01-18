import numpy as np
import pytest
import cirq
import cirq.testing
def test_density_matrix_invalid():
    bad_state = np.array([0.5, 0.5, 0.5])
    good_state = np.array([0.5, 0.5, 0.5, 0.5])
    with pytest.raises(ValueError):
        _ = cirq.density_matrix_from_state_vector(bad_state)
    with pytest.raises(ValueError):
        _ = cirq.density_matrix_from_state_vector(bad_state, [0, 1])
    with pytest.raises(IndexError):
        _ = cirq.density_matrix_from_state_vector(good_state, [-1, 0, 1])
    with pytest.raises(IndexError):
        _ = cirq.density_matrix_from_state_vector(good_state, [-1])