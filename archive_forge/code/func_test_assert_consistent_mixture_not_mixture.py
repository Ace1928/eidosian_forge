import pytest
import numpy as np
import cirq
def test_assert_consistent_mixture_not_mixture():
    not_mixture = cirq.amplitude_damp(0.1)
    with pytest.raises(AssertionError, match='has_mixture'):
        cirq.testing.assert_consistent_mixture(not_mixture)