import itertools
import numpy as np
import pytest
import cirq
import cirq.testing
def test_measure_density_matrix_tensor_different_left_right_shape():
    with pytest.raises(ValueError, match='not equal'):
        cirq.measure_density_matrix(np.array([1, 0, 0, 0]).reshape((2, 2, 1, 1)), [1], qid_shape=(2, 1))