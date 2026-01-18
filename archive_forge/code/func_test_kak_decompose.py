import random
import numpy as np
import pytest
import cirq
from cirq import value
from cirq import unitary_eig
@pytest.mark.parametrize('unitary', [cirq.testing.random_unitary(4), cirq.unitary(cirq.IdentityGate(2)), cirq.unitary(cirq.SWAP), cirq.unitary(cirq.SWAP ** 0.25), cirq.unitary(cirq.ISWAP), cirq.unitary(cirq.CZ ** 0.5), cirq.unitary(cirq.CZ)])
def test_kak_decompose(unitary: np.ndarray):
    kak = cirq.kak_decomposition(unitary)
    circuit = cirq.Circuit(kak._decompose_(cirq.LineQubit.range(2)))
    np.testing.assert_allclose(cirq.unitary(circuit), unitary, atol=1e-06)
    assert len(circuit) == 5
    assert len(list(circuit.all_operations())) == 8