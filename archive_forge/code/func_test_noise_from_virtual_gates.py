import numpy as np
import pytest
import sympy
import cirq
from cirq.devices.noise_utils import PHYSICAL_GATE_TAG
from cirq.devices.thermal_noise_model import (
def test_noise_from_virtual_gates():
    q0, q1 = cirq.LineQubit.range(2)
    gate_durations = {cirq.ZPowGate: 25.0}
    heat_rate_GHz = {q1: 1e-05}
    cool_rate_GHz = {q0: 0.0001}
    model = ThermalNoiseModel(qubits={q0, q1}, gate_durations_ns=gate_durations, heat_rate_GHz=heat_rate_GHz, cool_rate_GHz=cool_rate_GHz, dephase_rate_GHz=None, require_physical_tag=True, skip_measurements=False)
    moment = cirq.Moment(cirq.Z(q0), cirq.Z(q1))
    assert model.noisy_moment(moment, system_qubits=[q0, q1]) == [moment]
    part_virtual_moment = cirq.Moment(cirq.Z(q0), cirq.Z(q1).with_tags(PHYSICAL_GATE_TAG))
    with pytest.raises(ValueError, match='all physical or all virtual'):
        _ = model.noisy_moment(part_virtual_moment, system_qubits=[q0, q1])
    model.require_physical_tag = False
    assert len(model.noisy_moment(moment, system_qubits=[q0, q1])) == 2