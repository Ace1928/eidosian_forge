from typing import Dict, List
import pytest
import numpy as np
import sympy
from google.protobuf import json_format
import cirq
import cirq_google as cg
from cirq_google.api import v2
from cirq_google.serialization.circuit_serializer import _SERIALIZER_NAME
def test_deserialize_invalid_gate_set():
    serializer = cg.CircuitSerializer()
    proto = v2.program_pb2.Program(language=v2.program_pb2.Language(gate_set='not_my_gate_set'), circuit=v2.program_pb2.Circuit(scheduling_strategy=v2.program_pb2.Circuit.MOMENT_BY_MOMENT, moments=[]))
    with pytest.raises(ValueError, match='not_my_gate_set'):
        serializer.deserialize(proto)
    proto.language.gate_set = ''
    with pytest.raises(ValueError, match='Missing gate set'):
        serializer.deserialize(proto)
    proto = v2.program_pb2.Program(circuit=v2.program_pb2.Circuit(scheduling_strategy=v2.program_pb2.Circuit.MOMENT_BY_MOMENT, moments=[]))
    with pytest.raises(ValueError, match='Missing gate set'):
        serializer.deserialize(proto)