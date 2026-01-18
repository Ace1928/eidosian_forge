import itertools
import numpy as np
import pytest
import cirq
import cirq.testing
def test_measure_density_matrix_computational_basis_reversed():
    results = []
    for x in range(8):
        matrix = cirq.to_valid_density_matrix(x, 3)
        bits, out_matrix = cirq.measure_density_matrix(matrix, [0, 1, 2])
        results.append(bits)
        np.testing.assert_almost_equal(out_matrix, matrix)
    expected = [list(x) for x in list(itertools.product([False, True], repeat=3))]
    assert results == expected