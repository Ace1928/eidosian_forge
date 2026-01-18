import random
import numpy as np
import pytest
import cirq
from cirq import value
from cirq import unitary_eig
def test_kak_decomposition_invalid_object():
    with pytest.raises(TypeError, match='unitary effect'):
        _ = cirq.kak_decomposition('test')
    with pytest.raises(ValueError, match='4x4 unitary matrix'):
        _ = cirq.kak_decomposition(np.eye(3))
    with pytest.raises(ValueError, match='4x4 unitary matrix'):
        _ = cirq.kak_decomposition(np.eye(8))
    with pytest.raises(ValueError, match='4x4 unitary matrix'):
        _ = cirq.kak_decomposition(np.ones((4, 4)))
    with pytest.raises(ValueError, match='4x4 unitary matrix'):
        _ = cirq.kak_decomposition(np.zeros((4, 4)))
    nil = cirq.kak_decomposition(np.zeros((4, 4)), check_preconditions=False)
    np.testing.assert_allclose(cirq.unitary(nil), np.eye(4), atol=1e-08)