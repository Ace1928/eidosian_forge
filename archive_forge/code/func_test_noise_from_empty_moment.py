import numpy as np
import pytest
import sympy
import cirq
from cirq.devices.noise_utils import PHYSICAL_GATE_TAG
from cirq.devices.thermal_noise_model import (
def test_noise_from_empty_moment():
    q0, q1 = cirq.LineQubit.range(2)
    gate_durations = {}
    heat_rate_GHz = {q1: 1e-05}
    cool_rate_GHz = {q0: 0.0001}
    model = ThermalNoiseModel(qubits={q0, q1}, gate_durations_ns=gate_durations, heat_rate_GHz=heat_rate_GHz, cool_rate_GHz=cool_rate_GHz, dephase_rate_GHz=None, require_physical_tag=False, skip_measurements=False)
    moment = cirq.Moment()
    assert model.noisy_moment(moment, system_qubits=[q0, q1]) == [moment]