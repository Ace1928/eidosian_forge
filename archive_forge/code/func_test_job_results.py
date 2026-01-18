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
def test_job_results(client_constructor):
    grpc_client = _setup_client_mock(client_constructor)
    result = quantum.QuantumResult(parent='projects/proj/programs/prog/jobs/job0')
    grpc_client.get_quantum_result.return_value = result
    client = EngineClient()
    assert client.get_job_results('proj', 'prog', 'job0') == result
    grpc_client.get_quantum_result.assert_called_with(quantum.GetQuantumResultRequest(parent='projects/proj/programs/prog/jobs/job0'))