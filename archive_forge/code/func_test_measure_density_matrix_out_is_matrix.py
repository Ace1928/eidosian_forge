import itertools
import numpy as np
import pytest
import cirq
import cirq.testing
def test_measure_density_matrix_out_is_matrix():
    matrix = matrix_000_plus_010()
    bits, out_matrix = cirq.measure_density_matrix(matrix, [2, 1, 0], out=matrix)
    expected_state = np.zeros(8, dtype=np.complex64)
    expected_state[2 if bits[1] else 0] = 1.0
    expected_matrix = np.outer(np.conj(expected_state), expected_state)
    np.testing.assert_array_almost_equal(out_matrix, expected_matrix)
    assert out_matrix is matrix