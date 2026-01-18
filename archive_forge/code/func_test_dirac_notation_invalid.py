import numpy as np
import pytest
import cirq
import cirq.testing
def test_dirac_notation_invalid():
    with pytest.raises(ValueError, match='state_vector has incorrect size'):
        _ = cirq.dirac_notation([0.0, 0.0, 1.0])
    with pytest.raises(ValueError, match='state_vector has incorrect size'):
        _ = cirq.dirac_notation([1.0, 1.0], qid_shape=(3,))