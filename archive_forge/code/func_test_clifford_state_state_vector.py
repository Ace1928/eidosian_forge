import itertools
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_clifford_state_state_vector():
    q0, q1 = (cirq.LineQubit(0), cirq.LineQubit(1))
    state = cirq.CliffordState(qubit_map={q0: 0, q1: 1})
    np.testing.assert_equal(state.state_vector(), [1.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j])