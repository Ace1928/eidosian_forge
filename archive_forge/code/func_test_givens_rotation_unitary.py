import numpy as np
import pytest
import scipy
import sympy
import cirq
@pytest.mark.parametrize('angle_rads', (-np.pi, -np.pi / 3, -0.1, np.pi / 5))
def test_givens_rotation_unitary(angle_rads):
    actual = cirq.unitary(cirq.givens(angle_rads))
    c = np.cos(angle_rads)
    s = np.sin(angle_rads)
    expected = np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])
    assert np.allclose(actual, expected)