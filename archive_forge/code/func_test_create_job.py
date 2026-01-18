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
def test_create_job(client_constructor):
    grpc_client = _setup_client_mock(client_constructor)
    result = quantum.QuantumJob(name='projects/proj/programs/prog/jobs/job0')
    grpc_client.create_quantum_job.return_value = result
    run_context = any_pb2.Any()
    labels = {'hello': 'world'}
    client = EngineClient()
    assert client.create_job('proj', 'prog', 'job0', ['processor0'], run_context, 10, 'A job', labels) == ('job0', result)
    grpc_client.create_quantum_job.assert_called_with(quantum.CreateQuantumJobRequest(parent='projects/proj/programs/prog', quantum_job=quantum.QuantumJob(name='projects/proj/programs/prog/jobs/job0', run_context=run_context, scheduling_config=quantum.SchedulingConfig(priority=10, processor_selector=quantum.SchedulingConfig.ProcessorSelector(processor='projects/proj/processors/processor0', device_config_key=quantum.DeviceConfigKey(run_name='', config_alias=''))), description='A job', labels=labels)))
    assert client.create_job('proj', 'prog', 'job0', ['processor0'], run_context, 10, 'A job') == ('job0', result)
    grpc_client.create_quantum_job.assert_called_with(quantum.CreateQuantumJobRequest(parent='projects/proj/programs/prog', quantum_job=quantum.QuantumJob(name='projects/proj/programs/prog/jobs/job0', run_context=run_context, scheduling_config=quantum.SchedulingConfig(priority=10, processor_selector=quantum.SchedulingConfig.ProcessorSelector(processor='projects/proj/processors/processor0', device_config_key=quantum.DeviceConfigKey(run_name='', config_alias=''))), description='A job')))
    assert client.create_job('proj', 'prog', 'job0', ['processor0'], run_context, 10, labels=labels) == ('job0', result)
    grpc_client.create_quantum_job.assert_called_with(quantum.CreateQuantumJobRequest(parent='projects/proj/programs/prog', quantum_job=quantum.QuantumJob(name='projects/proj/programs/prog/jobs/job0', run_context=run_context, scheduling_config=quantum.SchedulingConfig(priority=10, processor_selector=quantum.SchedulingConfig.ProcessorSelector(processor='projects/proj/processors/processor0', device_config_key=quantum.DeviceConfigKey(run_name='', config_alias=''))), labels=labels)))
    assert client.create_job('proj', 'prog', 'job0', ['processor0'], run_context, 10) == ('job0', result)
    grpc_client.create_quantum_job.assert_called_with(quantum.CreateQuantumJobRequest(parent='projects/proj/programs/prog', quantum_job=quantum.QuantumJob(name='projects/proj/programs/prog/jobs/job0', run_context=run_context, scheduling_config=quantum.SchedulingConfig(priority=10, processor_selector=quantum.SchedulingConfig.ProcessorSelector(processor='projects/proj/processors/processor0', device_config_key=quantum.DeviceConfigKey(run_name='', config_alias=''))))))
    assert client.create_job('proj', 'prog', job_id=None, processor_ids=['processor0'], run_context=run_context, priority=10) == ('job0', result)
    grpc_client.create_quantum_job.assert_called_with(quantum.CreateQuantumJobRequest(parent='projects/proj/programs/prog', quantum_job=quantum.QuantumJob(run_context=run_context, scheduling_config=quantum.SchedulingConfig(priority=10, processor_selector=quantum.SchedulingConfig.ProcessorSelector(processor='projects/proj/processors/processor0', device_config_key=quantum.DeviceConfigKey(run_name='', config_alias=''))))))
    with pytest.raises(ValueError, match='priority must be between 0 and 1000'):
        client.create_job('proj', 'prog', job_id=None, processor_ids=['processor0'], run_context=run_context, priority=5000)