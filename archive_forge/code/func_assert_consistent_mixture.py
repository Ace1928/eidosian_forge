from typing import Any
import numpy as np
import cirq
def assert_consistent_mixture(gate: Any, rtol: float=1e-05, atol: float=1e-08):
    """Asserts that a given gate is a mixture and the mixture probabilities sum to one."""
    assert cirq.has_mixture(gate), f'Give gate {gate!r} does not return for cirq.has_mixture.'
    mixture = cirq.mixture(gate)
    total = np.sum(np.fromiter((k for k, v in mixture), dtype=float))
    assert np.abs(1 - total) <= atol + rtol * np.abs(total), f'The mixture for gate {gate!r} did not return coefficients that sum to 1. Summed to {total}.'