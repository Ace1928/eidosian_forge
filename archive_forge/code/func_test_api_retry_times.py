import asyncio
import datetime
import os
from unittest import mock
import duet
import pytest
from google.api_core import exceptions
from google.protobuf import any_pb2
from google.protobuf.field_mask_pb2 import FieldMask
from google.protobuf.timestamp_pb2 import Timestamp
from cirq_google.engine.engine_client import EngineClient, EngineException
import cirq_google.engine.stream_manager as engine_stream_manager
from cirq_google.cloud import quantum
@mock.patch('duet.sleep', return_value=duet.completed_future(None))
@mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
def test_api_retry_times(client_constructor, mock_sleep):
    grpc_client = _setup_client_mock(client_constructor)
    grpc_client.get_quantum_program.side_effect = exceptions.ServiceUnavailable('internal error')
    client = EngineClient(max_retry_delay_seconds=0.3)
    with pytest.raises(TimeoutError, match='Reached max retry attempts.*internal error'):
        client.get_program('proj', 'prog', False)
    assert grpc_client.get_quantum_program.call_count == 3
    assert len(mock_sleep.call_args_list) == 2
    assert all((x == y for (x, _), y in zip(mock_sleep.call_args_list, [(0.1,), (0.2,)])))