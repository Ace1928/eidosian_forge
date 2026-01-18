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
def test_get_device():
    processor = cg.EngineProcessor('a', 'p', EngineContext(), _processor=quantum.QuantumProcessor(device_spec=_DEVICE_SPEC))
    device = processor.get_device()
    assert device.metadata.qubit_set == frozenset([cirq.GridQubit(0, 0), cirq.GridQubit(1, 1), cirq.GridQubit(2, 2)])
    device.validate_operation(cirq.X(cirq.GridQubit(2, 2)))
    device.validate_operation(cirq.CZ(cirq.GridQubit(0, 0), cirq.GridQubit(1, 1)))
    with pytest.raises(ValueError):
        device.validate_operation(cirq.X(cirq.GridQubit(1, 2)))
    with pytest.raises(ValueError):
        device.validate_operation(cirq.H(cirq.GridQubit(0, 0)))
    with pytest.raises(ValueError):
        device.validate_operation(cirq.CZ(cirq.GridQubit(1, 1), cirq.GridQubit(2, 2)))