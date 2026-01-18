import random
import numpy as np
import pytest
import cirq
from cirq import value
from cirq import unitary_eig
@pytest.mark.parametrize('unitary,expected', ((np.eye(4), (0, 0, 0)), (SWAP, [np.pi / 4] * 3), (SWAP * 1j, [np.pi / 4] * 3), (CNOT, [np.pi / 4, 0, 0]), (CZ, [np.pi / 4, 0, 0]), (CZ @ SWAP, [np.pi / 4, np.pi / 4, 0]), (np.kron(X, X), (0, 0, 0))))
def test_KAK_vector_weyl_chamber_vertices(unitary, expected):
    actual = cirq.kak_vector(unitary)
    np.testing.assert_almost_equal(actual, expected)