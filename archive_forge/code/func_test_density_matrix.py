import numpy as np
import pytest
import cirq
import cirq.testing
def test_density_matrix():
    density_matrix_1 = np.eye(4, dtype=np.complex64) / 4
    state_vector_1 = cirq.one_hot(shape=(4,), dtype=np.complex64)
    state = cirq.density_matrix(density_matrix_1)
    assert state.data is density_matrix_1
    assert state.qid_shape == (2, 2)
    assert state.dtype == np.complex64
    with pytest.raises(ValueError, match='square'):
        _ = cirq.density_matrix(state_vector_1)