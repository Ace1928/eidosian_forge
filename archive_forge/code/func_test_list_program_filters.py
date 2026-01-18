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
@pytest.mark.parametrize('expected_filter, created_after, created_before, labels', [('', None, None, None), ('create_time >= 2020-09-01', datetime.date(2020, 9, 1), None, None), ('create_time >= 1598918400', datetime.datetime(2020, 9, 1, 0, 0, 0, tzinfo=datetime.timezone.utc), None, None), ('create_time <= 2020-10-01', None, datetime.date(2020, 10, 1), None), ('create_time >= 2020-09-01 AND create_time <= 1598918410', datetime.date(2020, 9, 1), datetime.datetime(2020, 9, 1, 0, 0, 10, tzinfo=datetime.timezone.utc), None), ('labels.color:red AND labels.shape:*', None, None, {'color': 'red', 'shape': '*'}), ('create_time >= 2020-08-01 AND create_time <= 1598918400 AND labels.color:red AND labels.shape:*', datetime.date(2020, 8, 1), datetime.datetime(2020, 9, 1, tzinfo=datetime.timezone.utc), {'color': 'red', 'shape': '*'})])
@mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
def test_list_program_filters(client_constructor, expected_filter, created_before, created_after, labels):
    grpc_client = _setup_client_mock(client_constructor)
    client = EngineClient()
    client.list_programs(project_id='proj', created_before=created_before, created_after=created_after, has_labels=labels)
    assert grpc_client.list_quantum_programs.call_args[0][0].filter == expected_filter