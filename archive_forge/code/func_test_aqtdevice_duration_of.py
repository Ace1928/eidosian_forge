from typing import List
import pytest
import cirq
from cirq_aqt.aqt_device_metadata import AQTDeviceMetadata
from cirq_aqt.aqt_target_gateset import AQTTargetGateset
def test_aqtdevice_duration_of(metadata, qubits):
    q0, q1 = qubits[:2]
    ms = cirq.Duration(millis=1)
    assert metadata.duration_of(cirq.Z(q0)) == 10 * ms
    assert metadata.duration_of(cirq.measure(q0)) == 100 * ms
    assert metadata.duration_of(cirq.measure(q0, q1)) == 100 * ms
    assert metadata.duration_of(cirq.XX(q0, q1)) == 200 * ms
    with pytest.raises(ValueError, match='Unsupported gate type'):
        metadata.duration_of(cirq.I(q0))