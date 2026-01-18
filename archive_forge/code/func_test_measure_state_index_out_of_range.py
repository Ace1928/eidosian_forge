import itertools
from typing import Optional
import pytest
import numpy as np
import cirq
import cirq.testing
from cirq import linalg
@pytest.mark.parametrize('use_np_transpose', [False, True])
def test_measure_state_index_out_of_range(use_np_transpose: bool):
    linalg.can_numpy_support_shape = lambda s: use_np_transpose
    state = cirq.to_valid_state_vector(0, 3)
    with pytest.raises(IndexError, match='-2'):
        cirq.measure_state_vector(state, [-2])
    with pytest.raises(IndexError, match='3'):
        cirq.measure_state_vector(state, [3])