import cmath
import numpy as np
import pytest
import cirq
from cirq.linalg import matrix_commutes
def test_is_orthogonal():
    assert cirq.is_orthogonal(np.empty((0, 0)))
    assert not cirq.is_orthogonal(np.empty((1, 0)))
    assert not cirq.is_orthogonal(np.empty((0, 1)))
    assert cirq.is_orthogonal(np.array([[1]]))
    assert cirq.is_orthogonal(np.array([[-1]]))
    assert not cirq.is_orthogonal(np.array([[1j]]))
    assert not cirq.is_orthogonal(np.array([[5]]))
    assert not cirq.is_orthogonal(np.array([[3j]]))
    assert not cirq.is_orthogonal(np.array([[1, 0]]))
    assert not cirq.is_orthogonal(np.array([[1], [0]]))
    assert not cirq.is_orthogonal(np.array([[1, 0], [0, -2]]))
    assert cirq.is_orthogonal(np.array([[1, 0], [0, -1]]))
    assert not cirq.is_orthogonal(np.array([[1j, 0], [0, 1]]))
    assert not cirq.is_orthogonal(np.array([[1, 0], [1, 1]]))
    assert not cirq.is_orthogonal(np.array([[1, 1], [0, 1]]))
    assert not cirq.is_orthogonal(np.array([[1, 1], [1, 1]]))
    assert not cirq.is_orthogonal(np.array([[1, -1], [1, 1]]))
    assert cirq.is_orthogonal(np.array([[1, -1], [1, 1]]) * np.sqrt(0.5))
    assert not cirq.is_orthogonal(np.array([[1, 1j], [1j, 1]]) * np.sqrt(0.5))
    assert not cirq.is_orthogonal(np.array([[1, -1j], [1j, 1]]) * np.sqrt(0.5))
    assert cirq.is_orthogonal(np.array([[1, 1e-11], [0, 1 + 1e-11]]))