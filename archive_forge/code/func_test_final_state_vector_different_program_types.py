import collections
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_final_state_vector_different_program_types():
    a, b = cirq.LineQubit.range(2)
    np.testing.assert_allclose(cirq.final_state_vector(cirq.X), [0, 1], atol=1e-08)
    ops = [cirq.H(a), cirq.CNOT(a, b)]
    np.testing.assert_allclose(cirq.final_state_vector(ops), [np.sqrt(0.5), 0, 0, np.sqrt(0.5)], atol=1e-08)
    np.testing.assert_allclose(cirq.final_state_vector(cirq.Circuit(ops)), [np.sqrt(0.5), 0, 0, np.sqrt(0.5)], atol=1e-08)