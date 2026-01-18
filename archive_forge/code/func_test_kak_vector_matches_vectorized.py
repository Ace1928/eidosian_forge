import random
import numpy as np
import pytest
import cirq
from cirq import value
from cirq import unitary_eig
def test_kak_vector_matches_vectorized():
    actual = cirq.kak_vector(_random_unitaries)
    expected = np.array([cirq.kak_vector(u) for u in _random_unitaries])
    np.testing.assert_almost_equal(actual, expected)