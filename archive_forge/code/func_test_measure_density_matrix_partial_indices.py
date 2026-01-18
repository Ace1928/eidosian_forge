import itertools
import numpy as np
import pytest
import cirq
import cirq.testing
def test_measure_density_matrix_partial_indices():
    for index in range(3):
        for x in range(8):
            matrix = cirq.to_valid_density_matrix(x, 3)
            bits, out_matrix = cirq.measure_density_matrix(matrix, [index])
            np.testing.assert_almost_equal(out_matrix, matrix)
            assert bits == [bool(1 & x >> 2 - index)]