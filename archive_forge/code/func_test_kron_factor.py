import random
import numpy as np
import pytest
import cirq
from cirq import value
from cirq import unitary_eig
@pytest.mark.parametrize('f1,f2', [(H, X), (H * 1j, X), (H, SQRT_X), (H, SQRT_SQRT_X), (H, H), (SQRT_SQRT_X, H), (X, np.eye(2)), (1j * X, np.eye(2)), (X, 1j * np.eye(2)), (-X, 1j * np.eye(2)), (X, X)] + [(cirq.testing.random_unitary(2), cirq.testing.random_unitary(2)) for _ in range(10)])
def test_kron_factor(f1, f2):
    p = cirq.kron(f1, f2)
    g, g1, g2 = cirq.kron_factor_4x4_to_2x2s(p)
    assert abs(np.linalg.det(g1) - 1) < 1e-05
    assert abs(np.linalg.det(g2) - 1) < 1e-05
    assert np.allclose(g * cirq.kron(g1, g2), p)
    assert_kronecker_factorization_within_tolerance(p, g, g1, g2)