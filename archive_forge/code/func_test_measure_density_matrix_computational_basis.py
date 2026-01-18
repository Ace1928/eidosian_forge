import itertools
import numpy as np
import pytest
import cirq
import cirq.testing
def test_measure_density_matrix_computational_basis():
    results = []
    for x in range(8):
        matrix = cirq.to_valid_density_matrix(x, 3)
        bits, out_matrix = cirq.measure_density_matrix(matrix, [2, 1, 0])
        results.append(bits)
        np.testing.assert_almost_equal(out_matrix, matrix)
    expected = [list(reversed(x)) for x in list(itertools.product([False, True], repeat=3))]
    assert results == expected