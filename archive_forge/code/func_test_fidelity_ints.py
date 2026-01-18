import numpy as np
import pytest
import cirq
def test_fidelity_ints():
    assert cirq.fidelity(3, 4) == 0.0
    assert cirq.fidelity(4, 4) == 1.0
    with pytest.raises(ValueError, match='non-negative'):
        _ = cirq.fidelity(-1, 2)
    with pytest.raises(ValueError, match='maximum'):
        _ = cirq.fidelity(4, 1, qid_shape=(2,))
    with pytest.raises(ValueError, match='maximum'):
        _ = cirq.fidelity(1, 4, qid_shape=(2,))