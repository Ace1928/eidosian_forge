from typing import Dict, List
import pytest
import numpy as np
import sympy
from google.protobuf import json_format
import cirq
import cirq_google as cg
from cirq_google.api import v2
from cirq_google.serialization.circuit_serializer import _SERIALIZER_NAME
def test_deserialize_schedule_not_supported():
    serializer = cg.CircuitSerializer()
    proto = v2.program_pb2.Program(language=v2.program_pb2.Language(gate_set=_SERIALIZER_NAME), schedule=v2.program_pb2.Schedule(scheduled_operations=[v2.program_pb2.ScheduledOperation(start_time_picos=0)]))
    with pytest.raises(ValueError, match='no longer supported'):
        serializer.deserialize(proto)