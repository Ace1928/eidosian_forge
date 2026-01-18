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
def test_calibration_defaults(get_job_results):
    qjob = quantum.QuantumJob(execution_status=quantum.ExecutionStatus(state=quantum.ExecutionStatus.State.SUCCESS), update_time=UPDATE_TIME)
    result = v2.calibration_pb2.FocusedCalibrationResult()
    result.results.add()
    get_job_results.return_value = quantum.QuantumResult(result=util.pack_any(result))
    job = cg.EngineJob('a', 'b', 'steve', EngineContext(), _job=qjob)
    data = job.calibration_results()
    get_job_results.assert_called_once_with('a', 'b', 'steve')
    assert len(data) == 1
    assert data[0].code == v2.calibration_pb2.CALIBRATION_RESULT_UNSPECIFIED
    assert data[0].error_message is None
    assert data[0].token is None
    assert data[0].valid_until is None
    assert len(data[0].metrics) == 0