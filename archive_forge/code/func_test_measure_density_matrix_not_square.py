import itertools
import numpy as np
import pytest
import cirq
import cirq.testing
def test_measure_density_matrix_not_square():
    with pytest.raises(ValueError, match='not square'):
        cirq.measure_density_matrix(np.array([1, 0, 0]), [1])
    with pytest.raises(ValueError, match='not square'):
        cirq.measure_density_matrix(np.array([1, 0, 0, 0]).reshape((2, 1, 2)), [1], qid_shape=(2, 1))