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
def test_get_missing_device():
    processor = cg.EngineProcessor('a', 'p', EngineContext(), _processor=quantum.QuantumProcessor())
    with pytest.raises(ValueError, match='device specification'):
        _ = processor.get_device()