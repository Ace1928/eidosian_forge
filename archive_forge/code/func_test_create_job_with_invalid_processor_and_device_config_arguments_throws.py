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
@mock.patch.dict(os.environ, clear='CIRQ_TESTING')
@mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
@pytest.mark.parametrize('processor_ids, processor_id, run_name, device_config_name, error_message', [(['processor0'], '', 'RUN_NAME', 'CONFIG_ALIAS', 'Cannot specify `run_name` or `device_config_name` if `processor_id` is empty'), (['processor0', 'processor1'], '', '', '', 'The use of multiple processors is no longer supported.'), (None, '', '', '', 'Must specify a processor id when creating a job.'), (None, 'processor0', 'RUN_NAME', '', 'Cannot specify only one of `run_name` and `device_config_name`'), (None, 'processor0', '', 'CONFIG_ALIAS', 'Cannot specify only one of `run_name` and `device_config_name`')])
def test_create_job_with_invalid_processor_and_device_config_arguments_throws(client_constructor, processor_ids, processor_id, run_name, device_config_name, error_message):
    grpc_client = _setup_client_mock(client_constructor)
    result = quantum.QuantumJob(name='projects/proj/programs/prog/jobs/job0')
    grpc_client.create_quantum_job.return_value = result
    client = EngineClient()
    with pytest.raises(ValueError, match=error_message):
        client.create_job(project_id='proj', program_id='prog', job_id=None, processor_ids=processor_ids, processor_id=processor_id, run_name=run_name, device_config_name=device_config_name)