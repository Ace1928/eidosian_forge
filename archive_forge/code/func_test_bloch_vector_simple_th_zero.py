import numpy as np
import pytest
import cirq
import cirq.testing
def test_bloch_vector_simple_th_zero():
    sqrt = np.sqrt(0.5)
    th_state = np.array([sqrt, 0.5 + 0.5j])
    bloch = cirq.bloch_vector_from_state_vector(th_state, 0)
    desired_simple = np.array([sqrt, sqrt, 0])
    np.testing.assert_array_almost_equal(bloch, desired_simple)