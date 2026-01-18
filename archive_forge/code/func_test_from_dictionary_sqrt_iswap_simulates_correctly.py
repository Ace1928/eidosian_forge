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
def test_from_dictionary_sqrt_iswap_simulates_correctly():
    parameters_ab = cirq_google.PhasedFSimCharacterization(theta=0.6, zeta=0.5, chi=0.4, gamma=0.3, phi=0.2)
    parameters_bc = cirq_google.PhasedFSimCharacterization(theta=0.8, zeta=-0.5, chi=-0.4, gamma=-0.3, phi=-0.2)
    parameters_cd_dict = {'theta': 0.1, 'zeta': 0.2, 'chi': 0.3, 'gamma': 0.4, 'phi': 0.5}
    a, b, c, d = cirq.LineQubit.range(4)
    circuit = cirq.Circuit([[cirq.X(a), cirq.Y(c)], [cirq.FSimGate(np.pi / 4, 0.0).on(a, b), cirq.FSimGate(np.pi / 4, 0.0).on(d, c)], [cirq.FSimGate(np.pi / 4, 0.0).on(b, c)], [cirq.FSimGate(np.pi / 4, 0.0).on(a, b), cirq.FSimGate(np.pi / 4, 0.0).on(c, d)]])
    expected_circuit = cirq.Circuit([[cirq.X(a), cirq.X(c)], [cirq.PhasedFSimGate(**parameters_ab.asdict()).on(a, b), cirq.PhasedFSimGate(**parameters_cd_dict).on(c, d)], [cirq.PhasedFSimGate(**parameters_bc.asdict()).on(b, c)], [cirq.PhasedFSimGate(**parameters_ab.asdict()).on(a, b), cirq.PhasedFSimGate(**parameters_cd_dict).on(c, d)]])
    engine_simulator = PhasedFSimEngineSimulator.create_from_dictionary_sqrt_iswap(parameters={(a, b): parameters_ab, (b, c): parameters_bc, (c, d): parameters_cd_dict})
    actual = engine_simulator.final_state_vector(circuit)
    expected = cirq.final_state_vector(expected_circuit)
    assert cirq.allclose_up_to_global_phase(actual, expected)