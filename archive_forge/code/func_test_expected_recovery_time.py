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
def test_expected_recovery_time():
    processor = cg.EngineProcessor('a', 'p', EngineContext(), _processor=quantum.QuantumProcessor())
    assert not processor.expected_recovery_time()
    processor = cg.EngineProcessor('a', 'p', EngineContext(), _processor=quantum.QuantumProcessor(expected_recovery_time=Timestamp(seconds=1581515101)))
    assert processor.expected_recovery_time() == datetime.datetime(2020, 2, 12, 13, 45, 1, tzinfo=datetime.timezone.utc)