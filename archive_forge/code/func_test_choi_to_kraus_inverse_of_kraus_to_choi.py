from typing import Iterable, Sequence
import numpy as np
import pytest
import cirq
@pytest.mark.parametrize('choi', (np.eye(4), np.diag([1, 0, 0, 1]), np.diag([0.2, 0.3, 0.8, 0.7]), np.array([[1, 0, 1, 0], [0, 1, 0, -1], [1, 0, 1, 0], [0, -1, 0, 1]]), np.array([[0.8, 0, 0, 0.5], [0, 0.3, 0, 0], [0, 0, 0.2, 0], [0.5, 0, 0, 0.7]])))
def test_choi_to_kraus_inverse_of_kraus_to_choi(choi):
    """Verifies that cirq.kraus_to_choi(cirq.choi_to_kraus(.)) is identity on Choi matrices."""
    kraus = cirq.choi_to_kraus(choi)
    recovered_choi = cirq.kraus_to_choi(kraus)
    assert np.allclose(recovered_choi, choi)