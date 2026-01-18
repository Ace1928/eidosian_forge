from unittest import mock
import datetime
import duet
import pytest
import freezegun
import numpy as np
from google.protobuf.duration_pb2 import Duration
from google.protobuf.text_format import Merge
from google.protobuf.timestamp_pb2 import Timestamp
import cirq
import cirq_google as cg
from cirq_google.api import v2
from cirq_google.engine import util
from cirq_google.engine.engine import EngineContext
from cirq_google.cloud import quantum
def test_get_device_specification():
    processor = cg.EngineProcessor('a', 'p', EngineContext(), _processor=quantum.QuantumProcessor())
    assert processor.get_device_specification() is None
    expected = v2.device_pb2.DeviceSpecification()
    expected.valid_qubits.extend(['0_0', '1_1', '2_2'])
    target = expected.valid_targets.add()
    target.name = '2_qubit_targets'
    target.target_ordering = v2.device_pb2.TargetSet.SYMMETRIC
    new_target = target.targets.add()
    new_target.ids.extend(['0_0', '1_1'])
    gate = expected.valid_gates.add()
    gate.cz.SetInParent()
    gate.gate_duration_picos = 1000
    gate = expected.valid_gates.add()
    gate.phased_xz.SetInParent()
    processor = cg.EngineProcessor('a', 'p', EngineContext(), _processor=quantum.QuantumProcessor(device_spec=_DEVICE_SPEC))
    assert processor.get_device_specification() == expected