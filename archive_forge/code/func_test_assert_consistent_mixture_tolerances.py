import pytest
import numpy as np
import cirq
def test_assert_consistent_mixture_tolerances():
    mixture = _MixtureGate(0.1, 0.9 - 1e-05)
    cirq.testing.assert_consistent_mixture(mixture)
    with pytest.raises(AssertionError, match='sum to 1'):
        cirq.testing.assert_consistent_mixture(mixture, rtol=0, atol=1e-06)
    with pytest.raises(AssertionError, match='sum to 1'):
        cirq.testing.assert_consistent_mixture(mixture, rtol=1e-06, atol=0)