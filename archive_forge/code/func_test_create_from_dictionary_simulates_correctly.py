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
def test_create_from_dictionary_simulates_correctly():
    parameters_ab_1 = {'theta': 0.6, 'zeta': 0.5, 'chi': 0.4, 'gamma': 0.3, 'phi': 0.2}
    parameters_ab_2 = {'theta': 0.1, 'zeta': 0.2, 'chi': 0.3, 'gamma': 0.4, 'phi': 0.5}
    parameters_bc = {'theta': 0.8, 'zeta': -0.5, 'chi': -0.4, 'gamma': -0.3, 'phi': -0.2}
    parameters_cd = {'theta': 0.1, 'zeta': 0.2, 'chi': 0.3, 'gamma': 0.4, 'phi': 0.5}
    a, b, c, d = cirq.LineQubit.range(4)
    circuit = cirq.Circuit([[cirq.X(a), cirq.Y(b), cirq.Z(c), cirq.H(d)], [SQRT_ISWAP_INV_GATE.on(a, b), SQRT_ISWAP_INV_GATE.on(d, c)], [SQRT_ISWAP_INV_GATE.on(b, c)], [cirq_google.SYC.on(a, b), SQRT_ISWAP_INV_GATE.on(c, d)]])
    expected_circuit = cirq.Circuit([[cirq.X(a), cirq.Y(b), cirq.Z(c), cirq.H(d)], [cirq.PhasedFSimGate(**parameters_ab_1).on(a, b), cirq.PhasedFSimGate(**parameters_cd).on(c, d)], [cirq.PhasedFSimGate(**parameters_bc).on(b, c)], [cirq.PhasedFSimGate(**parameters_ab_2).on(a, b), cirq.PhasedFSimGate(**parameters_cd).on(c, d)]])
    engine_simulator = PhasedFSimEngineSimulator.create_from_dictionary(parameters={(a, b): {SQRT_ISWAP_INV_GATE: parameters_ab_1, cirq_google.SYC: parameters_ab_2}, (b, c): {SQRT_ISWAP_INV_GATE: parameters_bc}, (c, d): {SQRT_ISWAP_INV_GATE: parameters_cd}})
    actual = engine_simulator.final_state_vector(circuit)
    expected = cirq.final_state_vector(expected_circuit)
    assert cirq.allclose_up_to_global_phase(actual, expected)