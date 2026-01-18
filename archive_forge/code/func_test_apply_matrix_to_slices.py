import numpy as np
import pytest
import cirq
import cirq.testing
from cirq import linalg
def test_apply_matrix_to_slices():
    with pytest.raises(ValueError, match='out'):
        target = np.eye(2)
        _ = cirq.apply_matrix_to_slices(target=target, matrix=np.eye(2), out=target, slices=[0, 1])
    with pytest.raises(ValueError, match='shape'):
        target = np.eye(2)
        _ = cirq.apply_matrix_to_slices(target=target, matrix=np.eye(3), slices=[0, 1])
    np.testing.assert_allclose(cirq.apply_matrix_to_slices(target=np.array(range(5)), matrix=np.eye(0), slices=[]), np.array(range(5)))
    np.testing.assert_allclose(cirq.apply_matrix_to_slices(target=np.eye(4), matrix=np.array([[2, 3], [5, 7]]), slices=[1, 2]), np.array([[1, 0, 0, 0], [0, 2, 3, 0], [0, 5, 7, 0], [0, 0, 0, 1]]))
    np.testing.assert_allclose(cirq.apply_matrix_to_slices(target=np.eye(4), matrix=np.array([[2, 3], [5, 7]]), slices=[2, 1]), np.array([[1, 0, 0, 0], [0, 7, 5, 0], [0, 3, 2, 0], [0, 0, 0, 1]]))
    np.testing.assert_allclose(cirq.apply_matrix_to_slices(target=np.array(range(8)).reshape((2, 2, 2)), matrix=np.array([[0, 1], [1, 0]]), slices=[(0, slice(None), 0), (1, slice(None), 0)]).reshape((8,)), [4, 1, 6, 3, 0, 5, 2, 7])
    out = np.zeros(shape=(4,))
    actual = cirq.apply_matrix_to_slices(target=np.array([1, 2, 3, 4]), matrix=np.array([[2, 3], [5, 7]]), slices=[1, 2], out=out)
    assert actual is out
    np.testing.assert_allclose(actual, np.array([1, 13, 31, 4]))