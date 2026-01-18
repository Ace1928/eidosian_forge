import datetime
import pytest
from google.protobuf.timestamp_pb2 import Timestamp
from cirq_google.cloud import quantum
from cirq_google.engine.abstract_local_processor import AbstractLocalProcessor
def test_bad_reservation():
    p = NothingProcessor(processor_id='test')
    with pytest.raises(ValueError, match='after the start time'):
        _ = p.create_reservation(start_time=_time(2000000), end_time=_time(1000000))