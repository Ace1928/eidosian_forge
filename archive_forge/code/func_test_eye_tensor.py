import numpy as np
import pytest
import cirq
import cirq.testing
def test_eye_tensor():
    assert np.all(cirq.eye_tensor((), dtype=int) == np.array(1))
    assert np.all(cirq.eye_tensor((1,), dtype=int) == np.array([[1]]))
    assert np.all(cirq.eye_tensor((2,), dtype=int) == np.array([[1, 0], [0, 1]]))
    assert np.all(cirq.eye_tensor((2, 2), dtype=int) == np.array([[[[1, 0], [0, 0]], [[0, 1], [0, 0]]], [[[0, 0], [1, 0]], [[0, 0], [0, 1]]]]))
    assert np.all(cirq.eye_tensor((2, 3), dtype=int) == np.array([[[[1, 0, 0], [0, 0, 0]], [[0, 1, 0], [0, 0, 0]], [[0, 0, 1], [0, 0, 0]]], [[[0, 0, 0], [1, 0, 0]], [[0, 0, 0], [0, 1, 0]], [[0, 0, 0], [0, 0, 1]]]]))
    assert np.all(cirq.eye_tensor((3, 2), dtype=int) == np.array([[[[1, 0], [0, 0], [0, 0]], [[0, 1], [0, 0], [0, 0]]], [[[0, 0], [1, 0], [0, 0]], [[0, 0], [0, 1], [0, 0]]], [[[0, 0], [0, 0], [1, 0]], [[0, 0], [0, 0], [0, 1]]]]))