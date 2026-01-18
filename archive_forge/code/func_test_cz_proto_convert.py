import numpy as np
import pytest
import sympy
import cirq
import cirq_google.api.v1.programs as programs
from cirq_google.api.v1 import operations_pb2
def test_cz_proto_convert():
    gate = cirq.CZ ** sympy.Symbol('k')
    proto = operations_pb2.Operation(exp_11=operations_pb2.Exp11(target1=operations_pb2.Qubit(row=2, col=3), target2=operations_pb2.Qubit(row=3, col=4), half_turns=operations_pb2.ParameterizedFloat(parameter_key='k')))
    assert_proto_dict_convert(gate, proto, cirq.GridQubit(2, 3), cirq.GridQubit(3, 4))
    gate = cirq.CZ ** 0.5
    proto = operations_pb2.Operation(exp_11=operations_pb2.Exp11(target1=operations_pb2.Qubit(row=2, col=3), target2=operations_pb2.Qubit(row=3, col=4), half_turns=operations_pb2.ParameterizedFloat(raw=0.5)))
    assert_proto_dict_convert(gate, proto, cirq.GridQubit(2, 3), cirq.GridQubit(3, 4))