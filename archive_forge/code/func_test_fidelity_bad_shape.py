import numpy as np
import pytest
import cirq
def test_fidelity_bad_shape():
    with pytest.raises(ValueError, match='Invalid quantum state'):
        _ = cirq.fidelity(np.array([[[1.0]]]), np.array([[[1.0]]]), qid_shape=(1,))