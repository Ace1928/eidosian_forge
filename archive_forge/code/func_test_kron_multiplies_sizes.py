import pytest
import numpy as np
import cirq
def test_kron_multiplies_sizes():
    assert cirq.kron(np.array([1, 2])).shape == (1, 2)
    assert cirq.kron(np.array([1, 2]), shape_len=1).shape == (2,)
    assert cirq.kron(np.array([1, 2]), np.array([3, 4, 5]), shape_len=1).shape == (6,)
    assert cirq.kron(shape_len=0).shape == ()
    assert cirq.kron(shape_len=1).shape == (1,)
    assert cirq.kron(shape_len=2).shape == (1, 1)
    assert np.allclose(cirq.kron(1j, np.array([2, 3])), np.array([2j, 3j]))
    assert np.allclose(cirq.kron(), np.eye(1))
    assert np.allclose(cirq.kron(np.eye(1)), np.eye(1))
    assert np.allclose(cirq.kron(np.eye(2)), np.eye(2))
    assert np.allclose(cirq.kron(np.eye(1), np.eye(1)), np.eye(1))
    assert np.allclose(cirq.kron(np.eye(1), np.eye(2)), np.eye(2))
    assert np.allclose(cirq.kron(np.eye(2), np.eye(3)), np.eye(6))
    assert np.allclose(cirq.kron(np.eye(2), np.eye(3), np.eye(4)), np.eye(24))