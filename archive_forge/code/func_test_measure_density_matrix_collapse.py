import itertools
import numpy as np
import pytest
import cirq
import cirq.testing
def test_measure_density_matrix_collapse():
    matrix = matrix_000_plus_010()
    for _ in range(10):
        bits, out_matrix = cirq.measure_density_matrix(matrix, [2, 1, 0])
        assert bits in [[False, False, False], [False, True, False]]
        expected = np.zeros(8, dtype=np.complex64)
        if bits[1]:
            expected[2] = 1j
        else:
            expected[0] = 1
        expected_matrix = np.outer(np.conj(expected), expected)
        np.testing.assert_almost_equal(out_matrix, expected_matrix)
        assert out_matrix is not matrix
    for _ in range(10):
        bits, out_matrix = cirq.measure_density_matrix(matrix, [2])
        np.testing.assert_almost_equal(out_matrix, matrix)
        assert bits == [False]
        bits, out_matrix = cirq.measure_density_matrix(matrix, [0])
        np.testing.assert_almost_equal(out_matrix, matrix)
        assert bits == [False]