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
@pytest.mark.parametrize('expected_filter, created_after, created_before, labels, execution_states, executed_processor_ids, scheduled_processor_ids, ', [('', None, None, None, None, None, None), ('create_time >= 2020-09-01', datetime.date(2020, 9, 1), None, None, None, None, None), ('create_time >= 1598918400', datetime.datetime(2020, 9, 1, 0, 0, 0, tzinfo=datetime.timezone.utc), None, None, None, None, None), ('create_time <= 2020-10-01', None, datetime.date(2020, 10, 1), None, None, None, None), ('create_time >= 2020-09-01 AND create_time <= 1598918410', datetime.date(2020, 9, 1), datetime.datetime(2020, 9, 1, 0, 0, 10, tzinfo=datetime.timezone.utc), None, None, None, None), ('labels.color:red AND labels.shape:*', None, None, {'color': 'red', 'shape': '*'}, None, None, None), ('(execution_status.state = FAILURE OR execution_status.state = CANCELLED)', None, None, None, [quantum.ExecutionStatus.State.FAILURE, quantum.ExecutionStatus.State.CANCELLED], None, None), ('create_time >= 2020-08-01 AND create_time <= 1598918400 AND labels.color:red AND labels.shape:* AND (execution_status.state = SUCCESS)', datetime.date(2020, 8, 1), datetime.datetime(2020, 9, 1, tzinfo=datetime.timezone.utc), {'color': 'red', 'shape': '*'}, [quantum.ExecutionStatus.State.SUCCESS], None, None), ('(executed_processor_id = proc1)', None, None, None, None, ['proc1'], None), ('(executed_processor_id = proc1 OR executed_processor_id = proc2)', None, None, None, None, ['proc1', 'proc2'], None), ('(scheduled_processor_ids: proc1)', None, None, None, None, None, ['proc1']), ('(scheduled_processor_ids: proc1 OR scheduled_processor_ids: proc2)', None, None, None, None, None, ['proc1', 'proc2'])])
@mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
def test_list_jobs_filters(client_constructor, expected_filter, created_before, created_after, labels, execution_states, executed_processor_ids, scheduled_processor_ids):
    grpc_client = _setup_client_mock(client_constructor)
    client = EngineClient()
    client.list_jobs(project_id='proj', program_id='prog', created_before=created_before, created_after=created_after, has_labels=labels, execution_states=execution_states, executed_processor_ids=executed_processor_ids, scheduled_processor_ids=scheduled_processor_ids)
    assert grpc_client.list_quantum_jobs.call_args[0][0].filter == expected_filter