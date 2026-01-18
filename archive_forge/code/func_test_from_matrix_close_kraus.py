import random
import numpy as np
import pytest
import sympy
import cirq
@pytest.mark.parametrize('unitary', [cirq.testing.random_unitary(2), cirq.testing.random_unitary(2), cirq.testing.random_unitary(2), np.array([[0, 1], [1j, 0]])])
def test_from_matrix_close_kraus(unitary: np.ndarray):
    gate = cirq.PhasedXZGate.from_matrix(unitary)
    kraus = cirq.kraus(gate)
    assert len(kraus) == 1
    cirq.testing.assert_allclose_up_to_global_phase(kraus[0], unitary, atol=1e-08)