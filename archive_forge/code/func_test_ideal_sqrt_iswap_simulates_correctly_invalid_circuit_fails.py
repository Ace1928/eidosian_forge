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
def test_ideal_sqrt_iswap_simulates_correctly_invalid_circuit_fails():
    engine_simulator = PhasedFSimEngineSimulator.create_with_ideal_sqrt_iswap()
    a, b = cirq.LineQubit.range(2)
    with pytest.raises(IncompatibleMomentError):
        circuit = cirq.Circuit([cirq.CZ.on(a, b)])
        engine_simulator.simulate(circuit)
    with pytest.raises(IncompatibleMomentError):
        circuit = cirq.Circuit(cirq.global_phase_operation(coefficient=1.0))
        engine_simulator.simulate(circuit)
    with pytest.raises(IncompatibleMomentError):
        circuit = cirq.Circuit(cirq.CircuitOperation(cirq.FrozenCircuit(cirq.X(a))))
        engine_simulator.simulate(circuit)