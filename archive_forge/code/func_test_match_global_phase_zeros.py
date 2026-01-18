import numpy as np
import pytest
import cirq
import cirq.testing
from cirq import linalg
def test_match_global_phase_zeros():
    z = np.array([[0, 0], [0, 0]])
    b = np.array([[1, 1], [1, 1]])
    z1, b1 = cirq.match_global_phase(z, b)
    np.testing.assert_allclose(z, z1, atol=1e-10)
    np.testing.assert_allclose(b, b1, atol=1e-10)
    z2, b2 = cirq.match_global_phase(z, b)
    np.testing.assert_allclose(z, z2, atol=1e-10)
    np.testing.assert_allclose(b, b2, atol=1e-10)
    z3, z4 = cirq.match_global_phase(z, z)
    np.testing.assert_allclose(z, z3, atol=1e-10)
    np.testing.assert_allclose(z, z4, atol=1e-10)