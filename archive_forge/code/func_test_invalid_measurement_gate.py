import numpy as np
import pytest
import sympy
import cirq
import cirq_google.api.v1.programs as programs
from cirq_google.api.v1 import operations_pb2
def test_invalid_measurement_gate():
    with pytest.raises(ValueError, match='length'):
        _ = programs.gate_to_proto(cirq.MeasurementGate(3, 'test', invert_mask=(True,)), (cirq.GridQubit(2, 3), cirq.GridQubit(3, 4)), delay=0)
    with pytest.raises(ValueError, match='no qubits'):
        _ = programs.gate_to_proto(cirq.MeasurementGate(1, 'test'), (), delay=0)