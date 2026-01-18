import cmath
import numpy as np
import pytest
import cirq
from cirq.linalg import matrix_commutes
def test_commutes_tolerance():
    atol = 0.5
    x = np.array([[0, 1], [1, 0]])
    z = np.array([[1, 0], [0, -1]])
    assert matrix_commutes(x, x + z * 0.1, atol=atol)
    assert not matrix_commutes(x, x + z * 0.5, atol=atol)