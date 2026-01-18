import pytest
import numpy as np
import cirq
def test_kron_spreads_values():
    u = np.array([[2, 3], [5, 7]])
    assert np.allclose(cirq.kron(np.eye(2), u), np.array([[2, 3, 0, 0], [5, 7, 0, 0], [0, 0, 2, 3], [0, 0, 5, 7]]))
    assert np.allclose(cirq.kron(u, np.eye(2)), np.array([[2, 0, 3, 0], [0, 2, 0, 3], [5, 0, 7, 0], [0, 5, 0, 7]]))
    assert np.allclose(cirq.kron(u, u), np.array([[4, 6, 6, 9], [10, 14, 15, 21], [10, 15, 14, 21], [25, 35, 35, 49]]))