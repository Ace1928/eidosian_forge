from typing import Iterable, Optional, Tuple
import collections
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq_google
from cirq_google.calibration.engine_simulator import (
from cirq_google.calibration import (
import cirq
def test_ideal_sqrt_iswap_simulates_correctly():
    a, b, c, d = cirq.LineQubit.range(4)
    circuit = cirq.Circuit([[cirq.X(a), cirq.Y(c)], [cirq.FSimGate(np.pi / 4, 0.0).on(a, b), cirq.FSimGate(np.pi / 4, 0.0).on(c, d)], [cirq.FSimGate(np.pi / 4, 0.0).on(b, c)]])
    engine_simulator = PhasedFSimEngineSimulator.create_with_ideal_sqrt_iswap()
    actual = engine_simulator.final_state_vector(circuit)
    expected = cirq.final_state_vector(circuit)
    assert cirq.allclose_up_to_global_phase(actual, expected)