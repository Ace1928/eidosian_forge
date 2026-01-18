import random
import numpy as np
import pytest
import cirq
from cirq import value
from cirq import unitary_eig
def test_KAK_vector_local_invariants_random_input():
    actual = _local_invariants_from_kak(cirq.kak_vector(_random_unitaries))
    expected = _local_invariants_from_kak(_kak_vecs)
    np.testing.assert_almost_equal(actual, expected)