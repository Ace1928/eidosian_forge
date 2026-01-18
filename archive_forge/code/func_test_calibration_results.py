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
def test_calibration_results(get_job_results):
    qjob = quantum.QuantumJob(execution_status=quantum.ExecutionStatus(state=quantum.ExecutionStatus.State.SUCCESS), update_time=UPDATE_TIME)
    get_job_results.return_value = CALIBRATION_RESULT
    job = cg.EngineJob('a', 'b', 'steve', EngineContext(), _job=qjob)
    data = job.calibration_results()
    get_job_results.assert_called_once_with('a', 'b', 'steve')
    assert len(data) == 1
    assert data[0].code == v2.calibration_pb2.ERROR_CALIBRATION_FAILED
    assert data[0].error_message == 'uh oh'
    assert data[0].token == 'abc'
    assert data[0].valid_until.timestamp() == 1234567891
    assert len(data[0].metrics)
    assert data[0].metrics['theta'] == {(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)): [0.9999]}