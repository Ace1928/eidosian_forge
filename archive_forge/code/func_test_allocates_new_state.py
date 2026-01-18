import itertools
import random
from typing import Type
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
def test_allocates_new_state():

    class NoUnitary(cirq.testing.SingleQubitGate):

        def _has_unitary_(self):
            return True

        def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs):
            return np.copy(args.target_tensor)
    q0 = cirq.LineQubit(0)
    simulator = cirq.Simulator()
    circuit = cirq.Circuit(NoUnitary()(q0))
    initial_state = np.array([np.sqrt(0.5), np.sqrt(0.5)], dtype=np.complex64)
    result = simulator.simulate(circuit, initial_state=initial_state)
    np.testing.assert_array_almost_equal(result.state_vector(), initial_state)
    assert not initial_state is result.state_vector()