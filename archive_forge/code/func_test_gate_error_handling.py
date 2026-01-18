import numpy as np
import cirq
import pytest
def test_gate_error_handling():
    with pytest.raises(ValueError, match='`target_state` must be a 1d numpy array.'):
        cirq.StatePreparationChannel(np.eye(2))
    with pytest.raises(ValueError, match='Matrix width \\(5\\) is not a power of 2'):
        cirq.StatePreparationChannel(np.ones(shape=5))