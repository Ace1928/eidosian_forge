import numpy as np
import pytest
import sympy
import cirq
from cirq.devices.noise_utils import PHYSICAL_GATE_TAG
from cirq.devices.thermal_noise_model import (
def test_noise_from_measurement():
    q0, q1 = cirq.LineQubit.range(2)
    gate_durations = {cirq.ZPowGate: 25.0, cirq.MeasurementGate: 4000.0}
    heat_rate_GHz = {q1: 1e-05}
    cool_rate_GHz = {q0: 0.0001}
    model = ThermalNoiseModel(qubits={q0, q1}, gate_durations_ns=gate_durations, heat_rate_GHz=heat_rate_GHz, cool_rate_GHz=cool_rate_GHz, dephase_rate_GHz=None, require_physical_tag=False, skip_measurements=True)
    moment = cirq.Moment(cirq.measure(q0, q1, key='m'))
    assert model.noisy_moment(moment, system_qubits=[q0, q1]) == [moment]
    part_measure_moment = cirq.Moment(cirq.measure(q0, key='m'), cirq.Z(q1))
    assert len(model.noisy_moment(part_measure_moment, system_qubits=[q0, q1])) == 2
    model.skip_measurements = False
    assert len(model.noisy_moment(moment, system_qubits=[q0, q1])) == 2