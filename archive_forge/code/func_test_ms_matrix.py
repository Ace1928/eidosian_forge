import numpy as np
import pytest
import sympy
import cirq
def test_ms_matrix():
    s = np.sqrt(0.5)
    np.testing.assert_allclose(cirq.unitary(cirq.ms(np.pi / 4)), np.array([[s, 0, 0, -1j * s], [0, s, -1j * s, 0], [0, -1j * s, s, 0], [-1j * s, 0, 0, s]]), atol=1e-08)
    np.testing.assert_allclose(cirq.unitary(cirq.ms(np.pi)), np.diag([-1, -1, -1, -1]), atol=1e-08)