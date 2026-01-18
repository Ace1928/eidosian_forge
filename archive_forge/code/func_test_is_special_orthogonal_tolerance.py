import cmath
import numpy as np
import pytest
import cirq
from cirq.linalg import matrix_commutes
def test_is_special_orthogonal_tolerance():
    atol = 0.5
    assert cirq.is_special_orthogonal(np.array([[1, 0], [-0.5, 1]]), atol=atol)
    assert not cirq.is_special_orthogonal(np.array([[1, 0], [-0.6, 1]]), atol=atol)
    assert cirq.is_special_orthogonal(np.array([[1.2, 0, 0], [0, 1.2, 0], [0, 0, 1 / 1.2]]), atol=atol)
    assert not cirq.is_special_orthogonal(np.array([[1.2, 0, 0], [0, 1.2, 0], [0, 0, 1.2]]), atol=atol)
    assert not cirq.is_special_orthogonal(np.array([[1.2, 0, 0], [0, 1.3, 0], [0, 0, 1 / 1.2]]), atol=atol)