import collections
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_final_state_vector_initial_state():
    np.testing.assert_allclose(cirq.final_state_vector(cirq.X, initial_state=0), [0, 1], atol=1e-08)
    np.testing.assert_allclose(cirq.final_state_vector(cirq.X, initial_state=1), [1, 0], atol=1e-08)
    np.testing.assert_allclose(cirq.final_state_vector(cirq.X, initial_state=[np.sqrt(0.5), 1j * np.sqrt(0.5)]), [1j * np.sqrt(0.5), np.sqrt(0.5)], atol=1e-08)