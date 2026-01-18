from typing import cast
import numpy as np
import pytest
import cirq
@pytest.mark.parametrize('use_protocol', [False, True])
@pytest.mark.parametrize('gate', [cirq.MeasurementGate(1, 'a'), cirq.MeasurementGate(1, 'a', invert_mask=(True,)), cirq.MeasurementGate(1, 'a', qid_shape=(3,)), cirq.MeasurementGate(2, 'a', invert_mask=(True, False), qid_shape=(2, 3))])
def test_measurement_with_key(use_protocol, gate):
    if use_protocol:
        gate1 = cirq.with_measurement_key_mapping(gate, {'a': 'b'})
    else:
        gate1 = gate.with_key('b')
    assert gate1.key == 'b'
    assert gate1.num_qubits() == gate.num_qubits()
    assert gate1.invert_mask == gate.invert_mask
    assert cirq.qid_shape(gate1) == cirq.qid_shape(gate)
    if use_protocol:
        gate2 = cirq.with_measurement_key_mapping(gate, {'a': 'a'})
    else:
        gate2 = gate.with_key('a')
    assert gate2 == gate