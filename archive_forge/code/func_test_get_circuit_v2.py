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
@mock.patch('cirq_google.engine.engine_client.EngineClient.get_program_async')
def test_get_circuit_v2(get_program_async):
    circuit = cirq.Circuit(cirq.X(cirq.GridQubit(5, 2)) ** 0.5, cirq.measure(cirq.GridQubit(5, 2), key='result'))
    program = cg.EngineProgram('a', 'b', EngineContext())
    get_program_async.return_value = quantum.QuantumProgram(code=_PROGRAM_V2)
    assert program.get_circuit() == circuit
    get_program_async.assert_called_once_with('a', 'b', True)