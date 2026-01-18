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
@mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
def test_api_doesnt_retry_not_found_errors(client_constructor):
    grpc_client = _setup_client_mock(client_constructor)
    grpc_client.get_quantum_program.side_effect = exceptions.NotFound('not found')
    client = EngineClient()
    with pytest.raises(EngineException, match='not found'):
        client.get_program('proj', 'prog', False)
    assert grpc_client.get_quantum_program.call_count == 1