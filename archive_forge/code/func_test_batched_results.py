import datetime
from unittest import mock
import pytest
import duet
from google.protobuf import any_pb2, timestamp_pb2
from google.protobuf.text_format import Merge
import cirq
import cirq_google as cg
from cirq_google.api import v1, v2
from cirq_google.engine import util
from cirq_google.cloud import quantum
from cirq_google.engine.engine import EngineContext
@mock.patch('cirq_google.engine.engine_client.EngineClient.get_job_results_async')
def test_batched_results(get_job_results):
    qjob = quantum.QuantumJob(execution_status=quantum.ExecutionStatus(state=quantum.ExecutionStatus.State.SUCCESS), update_time=UPDATE_TIME)
    get_job_results.return_value = BATCH_RESULTS
    job = cg.EngineJob('a', 'b', 'steve', EngineContext(), _job=qjob)
    data = job.results()
    assert len(data) == 4
    assert str(data[0]) == 'q=011'
    assert str(data[1]) == 'q=111'
    assert str(data[2]) == 'q=1101'
    assert str(data[3]) == 'q=1001'
    get_job_results.assert_called_once_with('a', 'b', 'steve')
    data = job.batched_results()
    assert len(data) == 2
    assert len(data[0]) == 2
    assert len(data[1]) == 2
    assert str(data[0][0]) == 'q=011'
    assert str(data[0][1]) == 'q=111'
    assert str(data[1][0]) == 'q=1101'
    assert str(data[1][1]) == 'q=1001'