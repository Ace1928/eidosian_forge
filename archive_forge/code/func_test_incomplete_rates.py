import numpy as np
import pytest
import sympy
import cirq
from cirq.devices.noise_utils import PHYSICAL_GATE_TAG
from cirq.devices.thermal_noise_model import (
def test_incomplete_rates():
    q0, q1 = cirq.LineQubit.range(2)
    gate_durations = {cirq.PhasedXZGate: 25.0}
    heat_rate_GHz = {q1: 1e-05}
    cool_rate_GHz = {q0: 0.0001}
    model = ThermalNoiseModel(qubits={q0, q1}, gate_durations_ns=gate_durations, heat_rate_GHz=heat_rate_GHz, cool_rate_GHz=cool_rate_GHz, dephase_rate_GHz=None)
    assert model.gate_durations_ns == gate_durations
    assert model.require_physical_tag
    assert model.skip_measurements
    assert np.allclose(model.rate_matrix_GHz[q0], np.array([[0, 0.0001], [0, 0]]))
    assert np.allclose(model.rate_matrix_GHz[q1], np.array([[0, 0], [1e-05, 0]]))