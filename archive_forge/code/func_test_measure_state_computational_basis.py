import itertools
from typing import Optional
import pytest
import numpy as np
import cirq
import cirq.testing
from cirq import linalg
@pytest.mark.parametrize('use_np_transpose', [False, True])
def test_measure_state_computational_basis(use_np_transpose: bool):
    linalg.can_numpy_support_shape = lambda s: use_np_transpose
    results = []
    for x in range(8):
        initial_state = cirq.to_valid_state_vector(x, 3)
        bits, state = cirq.measure_state_vector(initial_state, [2, 1, 0])
        results.append(bits)
        np.testing.assert_almost_equal(state, initial_state)
    expected = [list(reversed(x)) for x in list(itertools.product([False, True], repeat=3))]
    assert results == expected