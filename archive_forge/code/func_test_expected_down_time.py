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
@mock.patch('cirq_google.engine.engine_client.EngineClient.get_processor_async')
def test_expected_down_time(get_processor):
    processor = cg.EngineProcessor('a', 'p', EngineContext(), _processor=quantum.QuantumProcessor())
    assert not processor.expected_down_time()
    get_processor.return_value = quantum.QuantumProcessor(expected_down_time=Timestamp(seconds=1581515101))
    assert cg.EngineProcessor('a', 'p', EngineContext()).expected_down_time() == datetime.datetime(2020, 2, 12, 13, 45, 1, tzinfo=datetime.timezone.utc)
    get_processor.assert_called_once()