from datetime import timedelta
from typing import List
import pytest
import cirq
from cirq_aqt import aqt_device, aqt_device_metadata
def test_init_qubits(device, qubits):
    ms = cirq.Duration(millis=1)
    assert device.qubits == frozenset(qubits)
    with pytest.raises(TypeError, match='NamedQubit'):
        aqt_device.AQTDevice(measurement_duration=100 * ms, twoq_gates_duration=200 * ms, oneq_gates_duration=10 * ms, qubits=[cirq.LineQubit(0), cirq.NamedQubit('a')])