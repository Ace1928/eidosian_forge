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
@mock.patch('cirq_google.engine.engine_client.EngineClient.list_processors_async')
def test_list_processors(list_processors_async):
    processor1 = quantum.QuantumProcessor(name='projects/proj/processors/xmonsim')
    processor2 = quantum.QuantumProcessor(name='projects/proj/processors/gmonsim')
    list_processors_async.return_value = [processor1, processor2]
    result = cg.Engine(project_id='proj').list_processors()
    list_processors_async.assert_called_once_with('proj')
    assert [p.processor_id for p in result] == ['xmonsim', 'gmonsim']