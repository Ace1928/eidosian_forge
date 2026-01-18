import pytest
import sympy
import cirq
import cirq_google as cg
from cirq_google.api import v2
@pytest.mark.parametrize('repetitions', [1, 5, ['a', 'b', 'c']])
def test_circuit_op_to_proto(repetitions):
    serializer = cg.CircuitOpSerializer()
    if isinstance(repetitions, int):
        repetition_ids = None
    else:
        repetition_ids = repetitions
        repetitions = len(repetition_ids)
    to_serialize = cirq.CircuitOperation(circuit=default_circuit(), qubit_map={cirq.GridQubit(1, 1): cirq.GridQubit(1, 2)}, measurement_key_map={'m': 'results'}, param_resolver={'k': 1.0}, repetitions=repetitions, repetition_ids=repetition_ids)
    constants = [v2.program_pb2.Constant(string_value=DEFAULT_TOKEN), v2.program_pb2.Constant(circuit_value=default_circuit_proto())]
    raw_constants = {DEFAULT_TOKEN: 0, default_circuit(): 1}
    repetition_spec = v2.program_pb2.RepetitionSpecification()
    if repetition_ids is None:
        repetition_spec.repetition_count = repetitions
    else:
        for rep_id in repetition_ids:
            repetition_spec.repetition_ids.ids.append(rep_id)
    qubit_map = v2.program_pb2.QubitMapping()
    q_p1 = qubit_map.entries.add()
    q_p1.key.id = '1_1'
    q_p1.value.id = '1_2'
    measurement_key_map = v2.program_pb2.MeasurementKeyMapping()
    meas_p1 = measurement_key_map.entries.add()
    meas_p1.key.string_key = 'm'
    meas_p1.value.string_key = 'results'
    arg_map = v2.program_pb2.ArgMapping()
    arg_p1 = arg_map.entries.add()
    arg_p1.key.arg_value.string_value = 'k'
    arg_p1.value.arg_value.float_value = 1.0
    expected = v2.program_pb2.CircuitOperation(circuit_constant_index=1, repetition_specification=repetition_spec, qubit_map=qubit_map, measurement_key_map=measurement_key_map, arg_map=arg_map)
    actual = serializer.to_proto(to_serialize, constants=constants, raw_constants=raw_constants)
    assert actual == expected