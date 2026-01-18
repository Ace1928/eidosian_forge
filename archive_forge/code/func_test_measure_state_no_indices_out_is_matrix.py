import itertools
import numpy as np
import pytest
import cirq
import cirq.testing
def test_measure_state_no_indices_out_is_matrix():
    matrix = cirq.to_valid_density_matrix(0, 3)
    bits, out_matrix = cirq.measure_density_matrix(matrix, [], out=matrix)
    assert [] == bits
    np.testing.assert_almost_equal(out_matrix, matrix)
    assert out_matrix is matrix