import numpy as np
import pytest
import cirq
import cirq.testing
def test_bloch_vector_multi_big():
    five_qubit_plus_state = np.array([0.1767767] * 32)
    desired_simple = np.array([1, 0, 0])
    for qubit in range(5):
        bloch_i = cirq.bloch_vector_from_state_vector(five_qubit_plus_state, qubit)
        np.testing.assert_array_almost_equal(bloch_i, desired_simple)