import cmath
import numpy as np
import pytest
import cirq
from cirq.linalg import matrix_commutes
def test_is_diagonal():
    assert cirq.is_diagonal(np.empty((0, 0)))
    assert cirq.is_diagonal(np.empty((1, 0)))
    assert cirq.is_diagonal(np.empty((0, 1)))
    assert cirq.is_diagonal(np.array([[1]]))
    assert cirq.is_diagonal(np.array([[-1]]))
    assert cirq.is_diagonal(np.array([[5]]))
    assert cirq.is_diagonal(np.array([[3j]]))
    assert cirq.is_diagonal(np.array([[1, 0]]))
    assert cirq.is_diagonal(np.array([[1], [0]]))
    assert not cirq.is_diagonal(np.array([[1, 1]]))
    assert not cirq.is_diagonal(np.array([[1], [1]]))
    assert cirq.is_diagonal(np.array([[5j, 0], [0, 2]]))
    assert cirq.is_diagonal(np.array([[1, 0], [0, 1]]))
    assert not cirq.is_diagonal(np.array([[1, 0], [1, 1]]))
    assert not cirq.is_diagonal(np.array([[1, 1], [0, 1]]))
    assert not cirq.is_diagonal(np.array([[1, 1], [1, 1]]))
    assert not cirq.is_diagonal(np.array([[1, 0.1], [0.1, 1]]))
    assert cirq.is_diagonal(np.array([[1, 1e-11], [1e-10, 1]]))