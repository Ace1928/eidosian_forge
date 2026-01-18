import itertools
from typing import Optional
import pytest
import numpy as np
import cirq
import cirq.testing
from cirq import linalg
@pytest.mark.parametrize('use_np_transpose', [False, True])
def test_measure_state_empty_state(use_np_transpose: bool):
    linalg.can_numpy_support_shape = lambda s: use_np_transpose
    initial_state = np.array([1.0])
    bits, state = cirq.measure_state_vector(initial_state, [])
    assert [] == bits
    np.testing.assert_almost_equal(state, initial_state)