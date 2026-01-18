import datetime
from unittest import mock
import pytest
import numpy as np
from google.protobuf import any_pb2, timestamp_pb2
from google.protobuf.text_format import Merge
import cirq
import cirq_google as cg
from cirq_google.api import v1, v2
from cirq_google.engine import util
from cirq_google.cloud import quantum
from cirq_google.engine.engine import EngineContext
from cirq_google.engine.result_type import ResultType
@mock.patch('cirq_google.engine.engine_client.EngineClient.delete_job_async')
def test_delete_jobs(delete_job_async):
    program = cg.EngineProgram('a', 'b', EngineContext())
    program.delete_job('c')
    delete_job_async.assert_called_with('a', 'b', 'c')