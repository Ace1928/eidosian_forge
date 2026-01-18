from typing import Optional, List
import pytest
import cirq
from cirq.transformers.transformer_primitives import MAPPED_CIRCUIT_OP_TAG
@pytest.mark.parametrize('op_density', [0.1, 0.5, 0.9])
def test_merge_operations_complexity(op_density):
    prng = cirq.value.parse_random_state(11011)
    circuit = cirq.testing.random_circuit(20, 500, op_density, random_state=prng)
    for merge_func in [lambda _, __: None, lambda op1, _: op1, lambda _, op2: op2, lambda op1, op2: (op1, op2, None)[prng.choice(3)]]:

        def wrapped_merge_func(op1, op2):
            wrapped_merge_func.num_function_calls += 1
            return merge_func(op1, op2)
        wrapped_merge_func.num_function_calls = 0
        _ = cirq.merge_operations(circuit, wrapped_merge_func)
        total_operations = len([*circuit.all_operations()])
        assert wrapped_merge_func.num_function_calls <= 2 * total_operations