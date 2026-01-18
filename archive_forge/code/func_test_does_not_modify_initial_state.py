import itertools
import random
from typing import Type
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
def test_does_not_modify_initial_state():
    q0 = cirq.LineQubit(0)
    simulator = cirq.Simulator()

    class InPlaceUnitary(cirq.testing.SingleQubitGate):

        def _has_unitary_(self):
            return True

        def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs):
            args.target_tensor[0], args.target_tensor[1] = (args.target_tensor[1], args.target_tensor[0])
            return args.target_tensor
    circuit = cirq.Circuit(InPlaceUnitary()(q0))
    initial_state = np.array([1, 0], dtype=np.complex64)
    result = simulator.simulate(circuit, initial_state=initial_state)
    np.testing.assert_array_almost_equal(np.array([1, 0], dtype=np.complex64), initial_state)
    np.testing.assert_array_almost_equal(result.state_vector(), np.array([0, 1], dtype=np.complex64))