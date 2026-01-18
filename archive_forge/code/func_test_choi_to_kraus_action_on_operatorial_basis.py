from typing import Iterable, Sequence
import numpy as np
import pytest
import cirq
@pytest.mark.parametrize('choi', (np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]]), np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]), np.eye(4) / 2, np.diag([1, 0, 0, 1]), np.array([[0.6, 0.0, -0.1j, 0.1], [0.0, 0.1, 0.0, 0.1j], [0.1j, 0.0, 0.4, 0.0], [0.1, -0.1j, 0.0, 0.9]])))
def test_choi_to_kraus_action_on_operatorial_basis(choi):
    """Verifies that cirq.choi_to_kraus computes a valid Kraus representation."""
    kraus_operators = cirq.choi_to_kraus(choi)
    c = np.reshape(choi, (2, 2, 2, 2))
    for i in (0, 1):
        for j in (0, 1):
            input_rho = np.zeros((2, 2))
            input_rho[i, j] = 1
            actual_rho = apply_kraus_operators(kraus_operators, input_rho)
            expected_rho = c[:, i, :, j]
            assert np.allclose(actual_rho, expected_rho)