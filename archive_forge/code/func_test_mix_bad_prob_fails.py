import cirq
import numpy as np
import pytest
def test_mix_bad_prob_fails():
    mix = [(0.5, np.array([[1, 0], [0, 0]]))]
    with pytest.raises(ValueError, match='Unitary probabilities must sum to 1'):
        _ = cirq.MixedUnitaryChannel(mixture=mix, key='m')