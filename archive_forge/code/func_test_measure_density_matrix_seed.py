import itertools
import numpy as np
import pytest
import cirq
import cirq.testing
def test_measure_density_matrix_seed():
    n = 5
    matrix = np.eye(2 ** n) / 2 ** n
    bits, out_matrix1 = cirq.measure_density_matrix(matrix, range(n), seed=1234)
    assert bits == [False, False, True, True, False]
    bits, out_matrix2 = cirq.measure_density_matrix(matrix, range(n), seed=np.random.RandomState(1234))
    assert bits == [False, False, True, True, False]
    np.testing.assert_allclose(out_matrix1, out_matrix2)