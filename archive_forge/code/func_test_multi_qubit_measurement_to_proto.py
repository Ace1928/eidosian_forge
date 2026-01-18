import numpy as np
import pytest
import sympy
import cirq
import cirq_google.api.v1.programs as programs
from cirq_google.api.v1 import operations_pb2
def test_multi_qubit_measurement_to_proto():
    gate = cirq.MeasurementGate(2, 'test')
    proto = operations_pb2.Operation(measurement=operations_pb2.Measurement(targets=[operations_pb2.Qubit(row=2, col=3), operations_pb2.Qubit(row=3, col=4)], key='test'))
    assert_proto_dict_convert(gate, proto, cirq.GridQubit(2, 3), cirq.GridQubit(3, 4))