import pytest
import sympy
import cirq
import cirq_google as cg
from cirq_google.api import v2
def test_circuit_op_arg_key_errors():
    deserializer = cg.CircuitOpDeserializer()
    arg_map = v2.program_pb2.ArgMapping()
    p1 = arg_map.entries.add()
    p1.key.arg_value.float_value = 1.0
    p1.value.arg_value.float_value = 2.0
    serialized = v2.program_pb2.CircuitOperation(circuit_constant_index=1, arg_map=arg_map)
    constants = [v2.program_pb2.Constant(string_value=DEFAULT_TOKEN), v2.program_pb2.Constant(circuit_value=default_circuit_proto())]
    deserialized_constants = [DEFAULT_TOKEN, default_circuit()]
    with pytest.raises(ValueError, match='Invalid key parameter type'):
        deserializer.from_proto(serialized, constants=constants, deserialized_constants=deserialized_constants)