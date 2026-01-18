import itertools
import numpy as np
import pytest
import cirq
import cirq.testing
def test_measure_density_matrix_out_of_range():
    matrix = cirq.to_valid_density_matrix(0, 3)
    with pytest.raises(IndexError, match='-2'):
        cirq.measure_density_matrix(matrix, [-2])
    with pytest.raises(IndexError, match='3'):
        cirq.measure_density_matrix(matrix, [3])