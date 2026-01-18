from typing import Iterable, Sequence
import numpy as np
import pytest
import cirq
@pytest.mark.parametrize('choi', (np.eye(4), np.diag([1, 0, 0, 1]), np.diag([0.2, 0.3, 0.8, 0.7]), np.array([[1, 0, 1, 0], [0, 1, 0, -1], [1, 0, 1, 0], [0, -1, 0, 1]]), np.array([[0.8, 0, 0, 0.5], [0, 0.3, 0, 0], [0, 0, 0.2, 0], [0.5, 0, 0, 0.7]])))
def test_choi_to_superoperator_inverse_of_superoperator_to_choi(choi):
    superoperator = cirq.choi_to_superoperator(choi)
    recovered_choi = cirq.superoperator_to_choi(superoperator)
    assert np.allclose(recovered_choi, choi)
    recovered_superoperator = cirq.choi_to_superoperator(recovered_choi)
    assert np.allclose(recovered_superoperator, superoperator)