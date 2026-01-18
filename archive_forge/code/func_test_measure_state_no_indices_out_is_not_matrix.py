import itertools
import numpy as np
import pytest
import cirq
import cirq.testing
def test_measure_state_no_indices_out_is_not_matrix():
    matrix = cirq.to_valid_density_matrix(0, 3)
    out = np.zeros_like(matrix)
    bits, out_matrix = cirq.measure_density_matrix(matrix, [], out=out)
    assert [] == bits
    np.testing.assert_almost_equal(out_matrix, matrix)
    assert out is out_matrix
    assert out is not matrix