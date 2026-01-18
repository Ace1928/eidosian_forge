import datetime
from unittest import mock
import time
import numpy as np
import pytest
import duet
from google.protobuf import any_pb2, timestamp_pb2
from google.protobuf.text_format import Merge
import cirq
import cirq_google
import cirq_google as cg
from cirq_google.api import v1, v2
from cirq_google.engine import util
from cirq_google.cloud import quantum
from cirq_google.engine.engine import EngineContext
@mock.patch('cirq_google.engine.engine_client.EngineClient.get_processor')
def test_get_engine_device(get_processor):
    device_spec = util.pack_any(Merge('\nvalid_qubits: "0_0"\nvalid_qubits: "1_1"\nvalid_qubits: "2_2"\nvalid_targets {\n  name: "2_qubit_targets"\n  target_ordering: SYMMETRIC\n  targets {\n    ids: "0_0"\n    ids: "1_1"\n  }\n}\nvalid_gates {\n  gate_duration_picos: 1000\n  cz {\n  }\n}\nvalid_gates {\n  phased_xz {\n  }\n}\n', v2.device_pb2.DeviceSpecification()))
    get_processor.return_value = quantum.QuantumProcessor(device_spec=device_spec)
    device = cirq_google.get_engine_device('rainbow', 'project')
    assert device.metadata.qubit_set == frozenset([cirq.GridQubit(0, 0), cirq.GridQubit(1, 1), cirq.GridQubit(2, 2)])
    device.validate_operation(cirq.X(cirq.GridQubit(2, 2)))
    device.validate_operation(cirq.CZ(cirq.GridQubit(0, 0), cirq.GridQubit(1, 1)))
    with pytest.raises(ValueError):
        device.validate_operation(cirq.X(cirq.GridQubit(1, 2)))
    with pytest.raises(ValueError):
        device.validate_operation(cirq.H(cirq.GridQubit(0, 0)))
    with pytest.raises(ValueError):
        device.validate_operation(cirq.CZ(cirq.GridQubit(1, 1), cirq.GridQubit(2, 2)))