import pytest
import numpy as np
import cirq
def test_acts_like_kron_multiplies_sizes():
    assert np.allclose(cirq.kron_with_controls(), np.eye(1))
    assert np.allclose(cirq.kron_with_controls(np.eye(2), np.eye(3), np.eye(4)), np.eye(24))
    u = np.array([[2, 3], [5, 7]])
    assert np.allclose(cirq.kron_with_controls(u, u), np.array([[4, 6, 6, 9], [10, 14, 15, 21], [10, 15, 14, 21], [25, 35, 35, 49]]))