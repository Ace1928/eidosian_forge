import itertools
from typing import Optional
import pytest
import numpy as np
import cirq
import cirq.testing
from cirq import linalg
@pytest.mark.parametrize('use_np_transpose', [False, True])
def test_measure_state_not_power_of_two(use_np_transpose: bool):
    linalg.can_numpy_support_shape = lambda s: use_np_transpose
    with pytest.raises(ValueError, match='3'):
        _, _ = cirq.measure_state_vector(np.array([1, 0, 0]), [1])
    with pytest.raises(ValueError, match='5'):
        cirq.measure_state_vector(np.array([0, 1, 0, 0, 0]), [1])