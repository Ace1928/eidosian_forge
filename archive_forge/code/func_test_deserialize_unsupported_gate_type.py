from typing import Dict, List
import pytest
import numpy as np
import sympy
from google.protobuf import json_format
import cirq
import cirq_google as cg
from cirq_google.api import v2
from cirq_google.serialization.circuit_serializer import _SERIALIZER_NAME
def test_deserialize_unsupported_gate_type():
    serializer = cg.CircuitSerializer()
    operation_proto = op_proto({'gate': {'id': 'no_pow'}, 'args': {'half_turns': {'arg_value': {'float_value': 0.125}}}, 'qubits': [{'id': '1_1'}]})
    proto = v2.program_pb2.Program(language=v2.program_pb2.Language(arg_function_language='', gate_set=_SERIALIZER_NAME), circuit=v2.program_pb2.Circuit(scheduling_strategy=v2.program_pb2.Circuit.MOMENT_BY_MOMENT, moments=[v2.program_pb2.Moment(operations=[operation_proto])]))
    with pytest.raises(ValueError, match='no_pow'):
        serializer.deserialize(proto)