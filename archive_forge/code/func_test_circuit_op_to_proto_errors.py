import pytest
import sympy
import cirq
import cirq_google as cg
from cirq_google.api import v2
def test_circuit_op_to_proto_errors():
    serializer = cg.CircuitOpSerializer()
    to_serialize = cirq.CircuitOperation(default_circuit())
    constants = [v2.program_pb2.Constant(string_value=DEFAULT_TOKEN), v2.program_pb2.Constant(circuit_value=default_circuit_proto())]
    raw_constants = {DEFAULT_TOKEN: 0, default_circuit(): 1}
    with pytest.raises(ValueError, match='Serializer expected CircuitOperation'):
        serializer.to_proto(v2.program_pb2.Operation(), constants=constants, raw_constants=raw_constants)
    bad_raw_constants = {cirq.FrozenCircuit(): 0}
    with pytest.raises(ValueError, match='Encountered a circuit not in the constants table'):
        serializer.to_proto(to_serialize, constants=constants, raw_constants=bad_raw_constants)
    with pytest.raises(ValueError, match='Cannot serialize repetitions of type'):
        serializer.to_proto(to_serialize ** sympy.Symbol('a'), constants=constants, raw_constants=raw_constants)