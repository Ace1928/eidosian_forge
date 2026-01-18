from typing import Iterable, Sequence
import numpy as np
import pytest
import cirq
@pytest.mark.parametrize('choi, expected_kraus', ((np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]]), (np.eye(2),)), (np.array([[1, 0, 0, -1j], [0, 0, 0, 0], [0, 0, 0, 0], [1j, 0, 0, 1]]), (np.diag([-1j, 1]),)), (np.array([[1, 1, 1, -1], [1, 1, 1, -1], [1, 1, 1, -1], [-1, -1, -1, 1]]) / 2, (np.array([[1, 1], [1, -1]]) / np.sqrt(2),)), (np.diag([1, 0, 0, 1]), (np.diag([1, 0]), np.diag([0, 1]))), (np.array([[1, 0, 0, 0.8], [0, 0.36, 0, 0], [0, 0, 0, 0], [0.8, 0, 0, 0.64]]), (np.diag([1, 0.8]), np.array([[0, 0.6], [0, 0]]))), (np.eye(4) / 2, (np.array([[np.sqrt(0.5), 0], [0, 0]]), np.array([[0, np.sqrt(0.5)], [0, 0]]), np.array([[0, 0], [np.sqrt(0.5), 0]]), np.array([[0, 0], [0, np.sqrt(0.5)]])))))
def test_choi_to_kraus_fixed_values(choi, expected_kraus):
    """Verifies that cirq.choi_to_kraus gives correct results on a few fixed inputs."""
    actual_kraus = cirq.choi_to_kraus(choi)
    assert len(actual_kraus) == len(expected_kraus)
    for i in (0, 1):
        for j in (0, 1):
            input_rho = np.zeros((2, 2))
            input_rho[i, j] = 1
            actual_rho = apply_kraus_operators(actual_kraus, input_rho)
            expected_rho = apply_kraus_operators(expected_kraus, input_rho)
            assert np.allclose(actual_rho, expected_rho)