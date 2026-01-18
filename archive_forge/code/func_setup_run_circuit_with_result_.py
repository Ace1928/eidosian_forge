import datetime
from unittest import mock
import time
import numpy as np
import pytest
import duet
from google.protobuf import any_pb2, timestamp_pb2
from google.protobuf.text_format import Merge
import cirq
import cirq_google
import cirq_google as cg
from cirq_google.api import v1, v2
from cirq_google.engine import util
from cirq_google.cloud import quantum
from cirq_google.engine.engine import EngineContext
def setup_run_circuit_with_result_(client, result):
    client().create_program_async.return_value = ('prog', quantum.QuantumProgram(name='projects/proj/programs/prog'))
    client().create_job_async.return_value = ('job-id', quantum.QuantumJob(name='projects/proj/programs/prog/jobs/job-id', execution_status={'state': 'READY'}))
    client().get_job_async.return_value = quantum.QuantumJob(execution_status={'state': 'SUCCESS'}, update_time=_DT)
    client().get_job_results_async.return_value = quantum.QuantumResult(result=result)
    stream_future = duet.AwaitableFuture()
    stream_future.try_set_result(quantum.QuantumResult(result=result))
    client().run_job_over_stream.return_value = stream_future